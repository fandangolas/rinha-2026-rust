package vectorize

import (
	"time"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
)

// Normalization holds the constants from normalization.json.
type Normalization struct {
	MaxAmount         float32 `json:"max_amount"`
	MaxInstallments   float32 `json:"max_installments"`
	AmountVsAvgRatio  float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes        float32 `json:"max_minutes"`
	MaxKm             float32 `json:"max_km"`
	MaxTxCount24h     float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmt float32 `json:"max_merchant_avg_amount"`
}

// MCCRisk maps merchant category codes to risk scores (0.0–1.0).
type MCCRisk map[string]float32

const defaultMCCRisk = float32(0.5)

// Request mirrors the POST /fraud-score payload.
type Request struct {
	ID          string      `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer    Customer    `json:"customer"`
	Merchant    Merchant    `json:"merchant"`
	Terminal    Terminal    `json:"terminal"`
	LastTx      *LastTx     `json:"last_transaction"`
}

type Transaction struct {
	Amount       float32 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTx struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float32 `json:"km_from_current"`
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func boolF(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

func isKnownMerchant(id string, known []string) bool {
	for _, m := range known {
		if m == id {
			return true
		}
	}
	return false
}

// Vectorize converts a transaction request into a 14-dimensional search vector.
// It is allocation-free in the hot path (no map construction for known_merchants).
func Vectorize(req *Request, norm *Normalization, mccRisk MCCRisk) search.Vector {
	var v search.Vector

	// dim 0 — normalized transaction amount
	v[0] = clamp(req.Transaction.Amount / norm.MaxAmount)

	// dim 1 — normalized installment count
	v[1] = clamp(float32(req.Transaction.Installments) / norm.MaxInstallments)

	// dim 2 — amount relative to customer average
	if req.Customer.AvgAmount > 0 {
		v[2] = clamp((req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio)
	}

	// dim 3, 4 — time features
	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	v[3] = float32(t.UTC().Hour()) / 23.0
	v[4] = float32(t.UTC().Weekday()) / 6.0

	// dim 5, 6 — recency / travel distance (sentinel -1 when no history)
	if req.LastTx == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		v[5] = clamp(minutes / norm.MaxMinutes)
		v[6] = clamp(req.LastTx.KmFromCurrent / norm.MaxKm)
	}

	// dim 7 — distance from home
	v[7] = clamp(req.Terminal.KmFromHome / norm.MaxKm)

	// dim 8 — transaction frequency in last 24h
	v[8] = clamp(float32(req.Customer.TxCount24h) / norm.MaxTxCount24h)

	// dim 9 — online transaction flag
	v[9] = boolF(req.Terminal.IsOnline)

	// dim 10 — card present flag
	v[10] = boolF(req.Terminal.CardPresent)

	// dim 11 — merchant unknown to customer
	v[11] = boolF(!isKnownMerchant(req.Merchant.ID, req.Customer.KnownMerchants))

	// dim 12 — MCC risk score
	if risk, ok := mccRisk[req.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = defaultMCCRisk
	}

	// dim 13 — normalized merchant average amount
	v[13] = clamp(req.Merchant.AvgAmount / norm.MaxMerchantAvgAmt)

	return v
}
