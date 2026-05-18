// test-direct.js — same load profile as test.js but hits the API directly
// on port 9998 (bypasses HAProxy). Used to measure HAProxy overhead.
import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('./test-data.json')).entries;
});

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['p(99)'],
    systemTags: ['status', 'method'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 250,
            gracefulStop: '10s',
            stages: [
                { duration: '120s', target: 900 },
            ],
        },
    },
};

export default function () {
    const idx = exec.scenario.iterationInTest;
    if (idx >= testData.length) return;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://localhost:9998/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1);
            else tpCount.add(1);
        } else {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

export function handleSummary(data) {
    const K = 1000;
    const T_MAX_MS = 1000;
    const P99_MIN_MS = 1;
    const P99_MAX_MS = 2000;
    const EPSILON_MIN = 0.001;
    const BETA = 300;
    const TX_CORTE = 0.15;

    const p99 = data.metrics.http_req_duration.values['p(99)'];
    const tp = data.metrics.tp_count ? data.metrics.tp_count.values.count : 0;
    const tn = data.metrics.tn_count ? data.metrics.tn_count.values.count : 0;
    const fp = data.metrics.fp_count ? data.metrics.fp_count.values.count : 0;
    const fn_ = data.metrics.fn_count ? data.metrics.fn_count.values.count : 0;
    const errs = data.metrics.error_count ? data.metrics.error_count.values.count : 0;
    const N = tp + tn + fp + fn_ + errs;
    const E = (fp * 1) + (fn_ * 3) + (errs * 5);
    const failures = fp + fn_ + errs;
    const epsilon = N > 0 ? E / N : 0;
    const failureRate = N > 0 ? failures / N : 0;

    let p99Score;
    if (p99 <= 0) p99Score = 0;
    else if (p99 > P99_MAX_MS) p99Score = -3000;
    else p99Score = K * Math.log10(T_MAX_MS / Math.max(p99, P99_MIN_MS));

    let detScore, rateComponent = 0, absolutePenalty = 0, cutTriggered = false;
    if (failureRate > TX_CORTE) {
        detScore = -3000;
        cutTriggered = true;
    } else {
        rateComponent = K * Math.log10(1 / Math.max(epsilon, EPSILON_MIN));
        absolutePenalty = -BETA * Math.log10(1 + E);
        detScore = rateComponent + absolutePenalty;
    }

    const result = {
        target: 'direct-api (no HAProxy)',
        p99: p99.toFixed(2) + 'ms',
        scoring: {
            breakdown: { fp, fn: fn_, tp, tn, http_errors: errs },
            failure_rate: +(failureRate * 100).toFixed(2) + '%',
            p99_score: +p99Score.toFixed(2),
            detection_score: +detScore.toFixed(2),
            final_score: +(p99Score + detScore).toFixed(2),
        },
    };

    console.log('\n' + JSON.stringify(result, null, 2));
    return { 'test/results-direct.json': JSON.stringify(result, null, 2) };
}
