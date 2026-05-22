use memmap2::MmapOptions;
use std::{cell::RefCell, cmp::Ordering, fs::File, slice};

const DIMS: usize = 14;
const MAGIC: &[u8; 4] = b"IVFX";
const QUANT_SCALE: f32 = 127.0;

pub struct Neighbor {
    pub is_fraud: bool,
}

pub struct Searcher {
    mmap: memmap2::Mmap,
    num_cents: usize,
    num_vecs: usize,
    probes: usize,
    cent_off: usize,
    offset_off: usize,
    vec_off: usize,
    label_off: usize,
}

unsafe impl Send for Searcher {}
unsafe impl Sync for Searcher {}

impl Searcher {
    pub fn load(path: &str, probes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < 32 {
            return Err("file too small for header".into());
        }
        if &mmap[0..4] != MAGIC {
            return Err(format!("bad magic: {:?}", &mmap[0..4]).into());
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into()?);
        if version != 1 {
            return Err(format!("unsupported version: {version}").into());
        }

        let num_vecs = u64::from_le_bytes(mmap[8..16].try_into()?) as usize;
        let num_cents = u32::from_le_bytes(mmap[16..20].try_into()?) as usize;
        let file_dims = u32::from_le_bytes(mmap[20..24].try_into()?) as usize;
        if file_dims != DIMS {
            return Err(format!("expected {DIMS} dims, got {file_dims}").into());
        }
        let default_probes = u32::from_le_bytes(mmap[24..28].try_into()?) as usize;
        let actual_probes = if probes > 0 { probes } else { default_probes };

        let cent_off = 32;
        let offset_off = cent_off + num_cents * DIMS * 4;
        let vec_off = offset_off + (num_cents + 1) * 8;
        let label_off = vec_off + num_vecs * DIMS;

        if mmap.len() < label_off + num_vecs {
            return Err(format!(
                "file too small: need {}, got {}",
                label_off + num_vecs,
                mmap.len()
            )
            .into());
        }

        Ok(Searcher {
            mmap,
            num_cents,
            num_vecs,
            probes: actual_probes,
            cent_off,
            offset_off,
            vec_off,
            label_off,
        })
    }

    pub fn search(&self, query: &[f32; DIMS], k: usize) -> Vec<Neighbor> {
        let cents: &[f32] = unsafe {
            slice::from_raw_parts(
                self.mmap[self.cent_off..].as_ptr() as *const f32,
                self.num_cents * DIMS,
            )
        };
        let offsets: &[u64] = unsafe {
            slice::from_raw_parts(
                self.mmap[self.offset_off..].as_ptr() as *const u64,
                self.num_cents + 1,
            )
        };
        let vecs_i8: &[i8] = unsafe {
            slice::from_raw_parts(
                self.mmap[self.vec_off..].as_ptr() as *const i8,
                self.num_vecs * DIMS,
            )
        };
        let labels = &self.mmap[self.label_off..self.label_off + self.num_vecs];

        let probes = self.probes.min(self.num_cents);

        let mut query_i8 = [0i8; DIMS];
        for i in 0..DIMS {
            query_i8[i] = quantize(query[i]);
        }

        BUF.with_borrow_mut(|buf| {
            buf.cent_dists.clear();
            for i in 0..self.num_cents {
                let c = &cents[i * DIMS..(i + 1) * DIMS];
                let mut d = 0.0f32;
                for j in 0..DIMS {
                    let diff = query[j] - c[j];
                    d += diff * diff;
                }
                buf.cent_dists.push(CentDist { idx: i, dist: d });
            }

            if probes < buf.cent_dists.len() {
                buf.cent_dists.select_nth_unstable_by(probes - 1, |a, b| {
                    a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                });
            }
            buf.cent_dists[..probes].sort_unstable_by(|a, b| {
                a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
            });

            buf.cands.clear();

            for probe_i in 0..probes {
                let ci = buf.cent_dists[probe_i].idx;
                let start = offsets[ci] as usize;
                let end = offsets[ci + 1] as usize;

                for vi in start..end {
                    let vec_slice = &vecs_i8[vi * DIMS..(vi + 1) * DIMS];
                    let d = dist_int8(vec_slice, &query_i8);
                    let is_fraud = labels[vi] == 1;

                    if buf.cands.len() < k {
                        buf.cands.push(Candidate { dist: d, is_fraud });
                        if buf.cands.len() == k {
                            heapify_max(&mut buf.cands);
                        }
                    } else if d < buf.cands[0].dist {
                        buf.cands[0] = Candidate { dist: d, is_fraud };
                        sift_down_max(&mut buf.cands, 0);
                    }
                }
            }

            buf.cands
                .iter()
                .map(|c| Neighbor { is_fraud: c.is_fraud })
                .collect()
        })
    }
}

#[derive(Clone, Copy)]
struct CentDist {
    idx: usize,
    dist: f32,
}

#[derive(Clone, Copy)]
struct Candidate {
    dist: i32,
    is_fraud: bool,
}

struct SearchBuffer {
    cent_dists: Vec<CentDist>,
    cands: Vec<Candidate>,
}

thread_local! {
    static BUF: RefCell<SearchBuffer> = RefCell::new(SearchBuffer {
        cent_dists: Vec::with_capacity(5000),
        cands: Vec::with_capacity(8),
    });
}

#[inline(always)]
fn quantize(f: f32) -> i8 {
    let v = (f * QUANT_SCALE).round();
    if v > 127.0 {
        127
    } else if v < -127.0 {
        -127
    } else {
        v as i8
    }
}

#[inline(always)]
fn dist_int8(stored: &[i8], query: &[i8; DIMS]) -> i32 {
    let mut sum = 0i32;
    for i in 0..DIMS {
        let diff = stored[i] as i32 - query[i] as i32;
        sum += diff * diff;
    }
    sum
}

fn heapify_max(h: &mut [Candidate]) {
    let n = h.len();
    for i in (0..n / 2).rev() {
        sift_down_max(h, i);
    }
}

fn sift_down_max(h: &mut [Candidate], mut i: usize) {
    let n = h.len();
    loop {
        let mut largest = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n && h[l].dist > h[largest].dist {
            largest = l;
        }
        if r < n && h[r].dist > h[largest].dist {
            largest = r;
        }
        if largest == i {
            break;
        }
        h.swap(i, largest);
        i = largest;
    }
}
