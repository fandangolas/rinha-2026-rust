// Package uds exposes any Searcher over a Unix domain socket.
//
// Wire protocol (little-endian throughout):
//
//	Request (60 bytes, fixed):
//	  [0:56]  query  – 14 × float32
//	  [56:60] k      – int32
//
//	Response (4 + n×5 bytes):
//	  [0:4]   n      – int32  (actual neighbor count returned)
//	  per neighbor:
//	    [0:4] dist   – float32
//	    [4]   fraud  – uint8 (0=legit, 1=fraud)
package uds

import (
	"encoding/binary"
	"io"
	"log"
	"math"
	"net"
	"os"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
)

const (
	reqBytes      = search.Dims*4 + 4 // 60: query float32s + k int32
	respHeaderLen = 4
	neighborLen   = 5 // float32 distance + uint8 fraud flag
)

// Server wraps any Searcher and serves queries over a Unix domain socket.
// Safe to call Serve from a goroutine; Close shuts the listener and stops it.
type Server struct {
	ln net.Listener
	s  search.Searcher
}

// NewServer creates a listener at socketPath (removing any stale file first)
// and returns a Server ready for Serve to be called.
func NewServer(socketPath string, s search.Searcher) (*Server, error) {
	_ = os.Remove(socketPath)
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return nil, err
	}
	return &Server{ln: ln, s: s}, nil
}

// Serve accepts connections in a loop. Call from a goroutine; returns when the
// listener is closed.
func (srv *Server) Serve() {
	for {
		conn, err := srv.ln.Accept()
		if err != nil {
			return
		}
		go srv.handleConn(conn)
	}
}

// Close shuts the listener, causing Serve to return.
func (srv *Server) Close() error { return srv.ln.Close() }

func (srv *Server) handleConn(conn net.Conn) {
	defer conn.Close()
	var req [reqBytes]byte
	// response buffer reused across requests on this connection
	var respBuf [respHeaderLen + 128*neighborLen]byte

	for {
		if _, err := io.ReadFull(conn, req[:]); err != nil {
			return
		}

		var query search.Vector
		for i := range query {
			query[i] = math.Float32frombits(binary.LittleEndian.Uint32(req[i*4:]))
		}
		k := int(int32(binary.LittleEndian.Uint32(req[56:])))

		neighbors, err := srv.s.Search(query, k)
		if err != nil {
			log.Printf("uds server: search: %v", err)
			return
		}

		n := len(neighbors)
		respLen := respHeaderLen + n*neighborLen
		binary.LittleEndian.PutUint32(respBuf[0:], uint32(n))
		for i, nb := range neighbors {
			off := respHeaderLen + i*neighborLen
			binary.LittleEndian.PutUint32(respBuf[off:], math.Float32bits(nb.Distance))
			if nb.IsFraud {
				respBuf[off+4] = 1
			} else {
				respBuf[off+4] = 0
			}
		}
		if _, err := conn.Write(respBuf[:respLen]); err != nil {
			return
		}
	}
}
