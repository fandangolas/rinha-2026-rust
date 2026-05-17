package handler

import (
	"encoding/json"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/vectorize"
	"github.com/valyala/fasthttp"
)

// fraud_score is always k/5 where k ∈ {0,1,2,3,4,5}.
// Pre-compute all 6 response bodies to avoid any allocation or formatting
// in the hot path. Threshold is < 0.6, so fraud_score=0.6 (3/5) is NOT approved.
var cachedResponses = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1.0}`),
}

var contentTypeJSON = []byte("application/json")

// Handler holds the shared dependencies for HTTP request handling.
type Handler struct {
	searcher search.Searcher
	norm     *vectorize.Normalization
	mccRisk  vectorize.MCCRisk
}

// New creates a Handler. All fields must be non-nil.
func New(s search.Searcher, norm *vectorize.Normalization, mccRisk vectorize.MCCRisk) *Handler {
	return &Handler{searcher: s, norm: norm, mccRisk: mccRisk}
}

// Ready handles GET /ready.
func (h *Handler) Ready(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(fasthttp.StatusOK)
}

// FraudScore handles POST /fraud-score.
func (h *Handler) FraudScore(ctx *fasthttp.RequestCtx) {
	var req vectorize.Request
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	vec := vectorize.Vectorize(&req, h.norm, h.mccRisk)

	neighbors, err := h.searcher.Search(vec, 5)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusInternalServerError)
		return
	}

	fraudCount := countFraud(neighbors)
	ctx.SetContentTypeBytes(contentTypeJSON)
	ctx.SetBody(cachedResponses[fraudCount])
}

// Router returns a fasthttp.RequestHandler that dispatches by method + path.
func (h *Handler) Router() fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		switch {
		case ctx.IsGet() && string(ctx.Path()) == "/ready":
			h.Ready(ctx)
		case ctx.IsPost() && string(ctx.Path()) == "/fraud-score":
			h.FraudScore(ctx)
		default:
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	}
}

func countFraud(neighbors []search.Neighbor) int {
	n := 0
	for _, nb := range neighbors {
		if nb.IsFraud {
			n++
		}
	}
	return n
}
