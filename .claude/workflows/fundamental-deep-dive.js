export const meta = {
  name: 'fundamental-deep-dive',
  description: 'Test a stock thesis against primary-source SEC filings: five independent research lines, adversarial verification, then a verdict with falsifiable checkpoints.',
  whenToUse: 'When a holding or candidate needs a full fundamental workup — the user states a thesis and wants it tested, or a position is down and the story needs checking. Pass args: {symbol, cost, thesis, benchmark, sector, measured}. Expensive (~11 agents, ~45 min); for a quick screen run fund_metrics.py instead.',
  phases: [
    { title: 'Research', detail: 'five independent lines, all against primary sources' },
    { title: 'Verify', detail: 'adversarially attack the load-bearing claims' },
    { title: 'Synthesize', detail: 'verdict in Chinese + ledger registration' },
  ],
}

const A = (typeof args === 'object' && args) || {}
const SYMBOL = (A.symbol || 'AAPL').toUpperCase()
const COST = A.cost != null ? String(A.cost) : '未持有 / 未说明'
const THESIS = A.thesis || '(持有者未陈述论点 —— 先确立多头论点必须成立的条件，再检验它)'
const BENCH = A.benchmark || 'SPY'
const SECTOR = A.sector || '(未指定 —— 自行选择最有辩护力的行业 ETF 并说明理由)'
const MEASURED = A.measured || '(未提供已测量的技术/概率背景。若需要，自己用 move_prob.py 和 5×ATR22 止损惯例补齐)'

const ENV = `
THE POSITION AND THESIS UNDER TEST (this is what you are testing, not defending):
  Symbol: ${SYMBOL}          Holder cost: ${COST}
  Benchmark: ${BENCH}        Sector benchmark: ${SECTOR}
  Stated thesis: ${THESIS}

PRE-MEASURED TECHNICAL / PROBABILITY CONTEXT (do not redo, build on it):
${MEASURED}

YOUR JOB: establish what is TRUE about the fundamental thesis, from PRIMARY SOURCES. The holder
deserves to know whether their reasoning is supported by the company's actual economics, not whether
the story sounds good. A story that is factually true ("the product is everywhere", "everyone uses
it") can still be worthless as an investment argument if it is (a) already priced, or (b) not
connected to revenue the way the holder assumes. Those two failure modes are the common ones.

=== USE THE REPO'S DETERMINISTIC LAYER FOR ALL NUMBERS ===
Do NOT read financials out of a filing by eye. Layer 2 computes them from structured XBRL:
    cd /Users/feijing/github.com/stock_scan/Stock_OneClick/backend
    ../../vcp_env/bin/python _fund_data.py --symbols ${SYMBOL}
    ../../vcp_env/bin/python fund_metrics.py --symbols ${SYMBOL} --verbose
That yields revenue growth (annual and DATE-MATCHED quarterly YoY), margins, SBC/revenue, accruals,
dilution, RPO trend, capex intensity, related-party share of growth, EV/Sales and GAAP P/E — all
point-in-time. This split is not stylistic: an earlier workup asserted EV/Sales of 54.2x from a
stale hard-coded figure and only independent recomputation caught the true 48.8x. A model reading a
900k-character filing produces plausible numbers that are wrong.

Two XBRL traps that layer already handles, so do not re-introduce them by hand:
  * Quarterly duration facts are MISSING Q4 (the annual report covers the full year), so positional
    "a year ago" indexing silently reaches back 15 months. Match on date.
  * Tag migration: a company changes which us-gaap tag carries a concept and a single-tag lookup
    returns a truncated series. Apple's \`Revenues\` stops in 2018 for exactly this reason.

=== SOURCES: PRIMARY ONLY, AND ATTRIBUTE EVERY NUMBER ===
  * SEC EDGAR, free and keyless, but it REQUIRES a plausible contact User-Agent — a UA containing
    "@localhost" returns 403 on every endpoint while a real-looking address returns 200 on the
    identical request.
      https://data.sec.gov/submissions/CIK##########.json          (filing index)
      https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (structured facts)
      EDGAR full-text search for named-entity counts
  * FOREIGN PRIVATE ISSUERS file 20-F and 6-K, not 10-K/10-Q, and the fiscal year may not be the
    calendar year. Any filter on form == "10-K" drops them entirely. Check which forms this company
    files BEFORE concluding anything is missing.
  * The company's own IR releases and quarterly shareholder letters. When a fact appears in the
    filing notes but NOT in the shareholder letter investors actually read, that gap is itself a
    finding worth reporting.
  * Web search for context only. Every number carrying an argument must trace to a filing, an
    official IR document, or an exchange disclosure. Anything else is labelled "commentary,
    unverified". Never build an argument on a price target or an analyst opinion.

=== ENVIRONMENT ===
  Dir: /Users/feijing/github.com/stock_scan/Stock_OneClick/backend
  Python: /Users/feijing/github.com/stock_scan/vcp_env/bin/python  (numpy, pandas, yfinance,
  requests, lxml; NO scipy/sklearn/statsmodels). Save scripts as _fd_${SYMBOL}_<slug>.py and fetched
  documents as _fd_${SYMBOL}_<slug>.<ext> so the evidence is on disk and auditable.

=== RULES ===
  * LABEL EVERYTHING FACT / INFERENCE / SPECULATION. A FACT has a filing and an accession number.
    An INFERENCE has stated logic over facts. SPECULATION is permitted but must be named. Mixing
    them is the failure mode that makes an analysis worthless, because the reader cannot tell which
    parts to trust.
  * QUANTIFY OR DROP IT. "Customer X is important" is not a finding. "Customer X is at most the 12%
    slot, <=$590.4M, because the related-party residual is $4.0M" is.
  * WHEN IT IS NOT DISCLOSED, SAY SO, then derive the tightest bound. Per-unit economics and
    named-customer concentration are usually confidential. Never estimate one and present it as
    fact — the bound is the deliverable. The technique that works is the RESIDUAL TEST: total
    related-party revenue minus each identified related party equals the remainder; if the remainder
    is negligible, every other large customer slot must be external.
  * A STOCK THESIS IS ABOUT CHANGE IN EXPECTATIONS, NOT LEVEL OF QUALITY. Ubiquity that has been
    true for twenty years cannot explain a future return — the market has had two decades to price
    it. Hold every bullish claim to that standard. This is usually where a retail thesis breaks, and
    usually it is the whole error.
  * SEPARATE "good company" from "good stock at this price", and separate ESTIMATE CUTS from
    MULTIPLE COMPRESSION when explaining a drawdown. Those are completely different situations for
    a holder and the distinction is often the most decision-relevant thing you can produce.
  * SAMPLE SIZE. A recent IPO may have ~12 reported quarters. Any empirical claim must be compatible
    with that; say "n=3 cycles, cannot support a conclusion" rather than reporting a correlation as
    though it meant something.
  * The holder may be down and emotionally attached to the story. That is exactly why the analysis
    must be blunt. Do not soften a negative finding, and do not manufacture a positive one for
    balance.
`

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'What you established, with numbers and sources' },
    facts: { type: 'string', description: 'FACTS only — each with the primary source and accession number' },
    inferences: { type: 'string', description: 'What you infer from those facts, and the logic' },
    unknowable: { type: 'string', description: 'What is genuinely not disclosed, and the tightest bound you derived' },
    thesis_impact: { type: 'string', enum: ['supports', 'mixed', 'undermines', 'irrelevant'] },
    claims: {
      type: 'array',
      description: 'At most 2 load-bearing claims a skeptic should attack',
      items: {
        type: 'object',
        properties: {
          statement: { type: 'string' },
          evidence: { type: 'string' },
          strength: { type: 'string', enum: ['strong', 'moderate', 'weak'] },
        },
        required: ['statement', 'evidence', 'strength'],
      },
    },
  },
  required: ['summary', 'facts', 'inferences', 'unknowable', 'thesis_impact', 'claims'],
}

const DIMENSIONS = [
  {
    key: 'revenue-mechanics',
    prompt: `${ENV}

YOUR LINE: how does this company actually make money, and how would the holder's claimed catalyst
reach the income statement? This is the crux of most retail theses, so be forensic.

  1. REVENUE DECOMPOSITION from the filings: every disclosed segment and revenue type, by period,
     with growth rates. Which line is compounding and which is lumpy? Which line is the holder's
     thesis actually about?
  2. THE BILLING MECHANISM. Per unit, as a percentage of price, subscription, one-time licence? What
     is the disclosed average rate, if any? Critically: what is the LAG between the end-customer
     event and the company booking revenue? If revenue recognition lags a quarter or more, then a
     product launch or a design win is not a near-term catalyst at all, and the holder probably
     assumes it is.
  3. CUSTOMER CONCENTRATION, as precisely as the disclosures allow. Does the company disclose any
     customer >10% of revenue, named or unnamed? Apply the RESIDUAL TEST to pin down which slots
     must be external versus related-party. If the holder's thesis names a specific customer, bound
     that customer's contribution and state explicitly what is and is not disclosed.
  4. WHERE GROWTH ACTUALLY COMES FROM per the filings and IR commentary — rank segments by
     contribution to growth, not by size. It is common for the segment a holder cares about to be
     mature and contribute a minority of growth while something they never mention drives it. If
     that is the case here, say so with the split.
  5. Then compute the holder's implied elasticity: if the claimed driver grows +5%, what does that
     do to revenue, under the tightest bounds you can derive? Show the arithmetic, label every
     assumption, and stress-test the weakest step by overturning it entirely.`,
  },
  {
    key: 'valuation',
    prompt: `${ENV}

YOUR LINE: is the story already in the price? A true story that is fully priced earns nothing.

  1. REAL MULTIPLES at the current price, from filed numbers: trailing and forward P/E, price/sales,
     EV/sales, EV/gross profit, EV/EBITDA. State share count, net cash, and whether each figure is
     GAAP or adjusted. Cross-check against fund_metrics.py output rather than computing by hand.
  2. THE GAAP-VS-ADJUSTED GAP. Quantify share-based compensation as a percentage of revenue and its
     effect on margins. For any company whose bull case rests on margin expansion this is not a
     footnote — it is often the entire difference between the two narratives.
  3. PEER COMPARISON on an identical basis. Pick 3-4 genuine comparables and say why they are
     comparable. Where does this company rank on multiple versus growth?
  4. WHAT GROWTH IS IMPLIED. Reverse-engineer it: at the current price, what revenue CAGR and
     terminal margin are required at a defensible discount rate? Show the model and its assumptions.
     The question to answer in one sentence: does the price require a STEP-CHANGE, or merely a
     CONTINUATION of the demonstrated trend? Also price the continuation case explicitly so the
     holder can see the gap.
  5. THE DRAWDOWN QUESTION, if the stock is well off its high. Establish from the filings whether
     results and guidance actually deteriorated, or whether this was pure multiple compression.
     These are entirely different situations for a holder and this is often the single most useful
     thing you can deliver.
  6. Where does the holder's cost basis sit against every defensible valuation method you construct?
     Report it as a percentile of the stock's own trading history too.`,
  },
  {
    key: 'competitive-threat',
    prompt: `${ENV}

YOUR LINE: what could break the business model, and is it actually happening?

  1. THE STRUCTURAL THREAT. Identify the credible technological or competitive substitute, then find
     PRIMARY evidence of adoption rather than advocacy: named shipping products, company
     announcements, disclosed volumes, disclosed deal sizes. Distinguish "this exists and is loud"
     from "this is taking revenue". Quantify where you can and say "no reliable volume data" where
     you cannot — a clean negative is a valuable finding and stops the holder worrying about the
     wrong thing.
     Useful check: EDGAR full-text search for the substitute's name across the major customers'
     own filings. If the companies that would adopt it never mention it, that bounds the threat.
     State the coverage limit honestly: EDGAR indexes US registrants only, so private and foreign
     players being absent is a data gap, not evidence of absence.
  2. CUSTOMER CONFLICT. Is the company moving up the stack into what its own customers do? Establish
     the factual record of any dispute: what was claimed, what was decided, when, and what it
     implies for pricing power. A lost case that caps the company's ability to reprice its largest
     customer is a durable economic fact, not news flow.
  3. VERTICAL INTEGRATION BY CUSTOMERS. Are the large customers building in-house substitutes? If
     the industry trend moves customers toward arrangements that pay this company LESS per unit,
     that is a structural headwind and it often runs directly against a naive "everyone uses them so
     they win" thesis. Verify the direction from disclosures.
  4. GEOGRAPHIC AND GOVERNANCE RISK. Any jurisdiction where revenue is recognised through an entity
     the company does not control, or where amounts are self-reported by a counterparty? Quantify it
     as a percentage of revenue and give the disclosed history of any information or payment issues.
  5. OWNERSHIP STRUCTURE AND FLOAT. Verify the largest holder's stake, the actual free float, any
     lock-up status, registration rights, and pledged shares. A small float on a large market cap
     means narrative flows cannot be absorbed and repricing is violent. Be precise about the
     difference between a SUPPLY overhang and a FORCED-SALE risk — compute the loan-to-value if
     shares are pledged rather than implying a margin call is near.`,
  },
  {
    key: 'catalyst-empirics',
    prompt: `${ENV}

YOUR LINE: test the holder's causal claim EMPIRICALLY. Does the claimed driver actually move this
company's revenue or its stock? This is the one line that can be settled with data rather than
argument, so settle it.

Use the repo's Python environment and yfinance.
  1. BETA STRUCTURE — do this first, it is the cheapest decisive test. Regress this stock's daily
     returns on the claimed driver, on its sector ETF, on 2-3 obvious alternatives, and on the
     market, both separately and jointly. Report beta, R-squared and t-statistics. If the beta to
     the sector dwarfs the beta to the holder's claimed driver, and the driver's coefficient goes
     insignificant in the joint regression, then the market does not trade this stock as a
     derivative of that driver and the holder's mental model of the transmission channel is simply
     wrong. Say so with the numbers.
  2. EVENT STUDY on the claimed catalyst. Build the list of event dates from PRIMARY sources (the
     company's or the driver's own newsroom / filings, never memory). Measure abnormal returns
     around each event, raw and net of the sector, over [-5,+5] and [-1,+20] trading days. Report n
     and state plainly if n is too small to conclude anything — then widen to a related, more
     frequent event class (earnings dates) to get power.
  3. REVENUE-LEVEL TEST. Line up this company's reported quarterly revenue for the relevant segment
     against the driver's own reported figures. Compute the correlation and, more usefully, whether
     growth tracks growth. Give the actual series so the holder can see it, and report n.
  4. THE MATURITY CHECK. Get the end-market's unit growth from a citable source. If the end market
     is flat to low-single-digit, then the holder's catalyst is a replacement-cycle event rather
     than a growth event, and the revenue impact is bounded by content or price rather than units.
     Quantify that bound.
  5. Finally, and most usefully: what DOES explain this stock's recent move? Look for the proximate
     causes in filings, guidance changes and disclosed events in that window. If guidance was cut or
     a major customer event occurred, that is far more decision-relevant than the holder's catalyst.`,
  },
  {
    key: 'holder-decision',
    prompt: `${ENV}

YOUR LINE: frame the decision the holder actually faces, using this repo's own measured tools. You
are NOT giving investment advice — you are laying out the arithmetic of each option so the choice is
informed. Every number must come from a real computation, not a textbook formula.

  1. THE SUNK-COST FRAME. The cost basis is a fact about the holder's past, not about the company;
     the market does not know it. State that plainly, then show what it implies: the move required
     to break even, and the calibrated probability of still being below cost. Use the repo's own
     model:
        ../../vcp_env/bin/python move_prob.py ${SYMBOL} --thr 2
     and for cost-specific probabilities the same machinery as _pos_arm.py (P(close < cost) at each
     horizon, derived from the fitted sigma and the empirical z table). Reframe the real question as:
     "at today's price, would you buy it? If not, holding is buying."
  2. POSITION SIZING ARITHMETIC. From the model's 21-day 5th-percentile outcome and ATR22, compute
     the portfolio impact at 2% / 5% / 10% / 20% weights, and invert it: for a stated pain budget,
     what is the maximum position? The point is to reveal whether the holder's discomfort is about
     the stock or about the size — very often it is the size.
  3. THE STOP ARITHMETIC. Use the repo's 5×ATR22 trailing stop convention. Report the level, the
     distance in percent AND in ATR units, and the probability of TOUCHING it within 21 days
     (a touch probability is higher than a close-below probability — derive both and say which is
     which). If the stop sits only ~2 ATR away, say plainly that it is a coin flip on path noise
     rather than a risk control.
  4. THE OPTIONS ARITHMETIC with REAL quotes via yfinance. First validate the chain: check
     put-call parity implied forward against spot and report the error, so a garbage chain is caught
     before it is used. Then price a covered call, a protective put and a collar at real strikes with
     real premiums. Compare IV to realized vol and say whether you are selling rich vol or merely
     fair vol — that distinction decides whether an overwrite is compensated. Note the absolute cost
     when IV is high.
  5. WHAT WOULD MAKE THE THESIS CHECKABLE. 3-5 specific, DATED, falsifiable observations with a
     numeric threshold each. Start from the next scheduled earnings date. "Watch for weakness" is
     not acceptable; "segment revenue below $X, or y/y growth below Y%" is.`,
  },
]

phase('Research')
log(`Testing the ${SYMBOL} thesis along ${DIMENSIONS.length} independent lines, primary sources only`)

const lines = (await parallel(DIMENSIONS.map(d => () =>
  agent(d.prompt, { label: `research:${d.key}`, phase: 'Research', schema: SCHEMA })
    .then(r => r && ({ ...r, key: d.key }))
))).filter(Boolean)

log(`${lines.length}/${DIMENSIONS.length} returned; thesis impact: ` +
    lines.map(l => `${l.key}=${l.thesis_impact}`).join(', '))

// Barrier: the skeptics should attack the load-bearing claims across ALL lines, and the synthesis
// needs every line at once to weigh a thesis rather than list findings.
const claims = lines.flatMap(l => (l.claims || []).map(c => ({ ...c, from: l.key })))
  .sort((a, b) => ({ strong: 0, moderate: 1, weak: 2 }[a.strength] ?? 3) -
                  ({ strong: 0, moderate: 1, weak: 2 }[b.strength] ?? 3))
  .slice(0, 5)
log(`${claims.length} load-bearing claims to attack`)

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reasoning: { type: 'string' },
    corrected_statement: { type: 'string' },
  },
  required: ['refuted', 'confidence', 'reasoning', 'corrected_statement'],
}

phase('Verify')
const verdicts = (await parallel(claims.map(c => () =>
  agent(`${ENV}

ADVERSARIAL VERIFICATION. Attack this claim about ${SYMBOL}. Default to refuted=true when the
evidence is ambiguous. Go back to the PRIMARY SOURCE yourself — do not trust the claim's citation.

CLAIM (from the "${c.from}" line):
  "${c.statement}"
  Evidence offered: ${c.evidence}
  Self-rated strength: ${c.strength}

Attack it on:
 1. SOURCE INTEGRITY. Did the number come from a filing, or from commentary dressed as fact? Pull
    the source and check the figure, the period, and whether GAAP versus adjusted was switched
    mid-argument.
 2. IS IT DISCLOSED AT ALL? Per-unit rates and customer economics are often confidential. If the
    claim states one as fact, find out whether the company actually discloses it. An undisclosed
    number presented confidently is the most damaging error available here.
 3. DIRECTION OF THE ARGUMENT. Does the fact support the conclusion drawn from it, or could it
    equally support the opposite? Steelman the other reading before rejecting it.
 4. ALREADY PRICED. If the claim is bullish: is it new information, or a long-known feature of the
    business the market has had years to price?
 5. AGGREGATION. If the claim combines several items into a headline percentage, check each
    component belongs in that aggregate. A previous run reached "74% of growth from the parent's
    orbit" only by folding in an entity the same filing states is independently controlled; the
    defensible figure was 61%. Recompute aggregates from components.
 6. FORWARD REVERSAL. Check the MOST RECENT reported period. A quality-of-growth problem in the last
    annual report may already have reversed in the latest quarter, and a multiple capitalises forward
    growth, not trailing.
 7. SAMPLE SIZE / WINDOW. Check whether the claim's confidence is compatible with the number of
    reported periods available. Knock down small-n results stated firmly.

Write _fd_${SYMBOL}_verify_<slug>.py if you need to compute anything, and report what the sources say.`,
    { label: `verify:${c.from}`, phase: 'Verify', schema: VERDICT })
    .then(v => v && ({ claim: c, ...v }))
))).filter(Boolean)

log(`${verdicts.filter(v => !v.refuted).length}/${verdicts.length} claims survived`)

phase('Synthesize')
const synthesis = await agent(`${ENV}

Write the verdict on the holder's thesis, in CHINESE (the user reads Chinese; keep ticker symbols,
financial line-item names and units in English where there is no clean translation).

=== THE FIVE RESEARCH LINES ===
${lines.map(l => `--- ${l.key}  [对论点的影响: ${l.thesis_impact}] ---
${l.summary}

事实(FACTS): ${l.facts}

推断(INFERENCES): ${l.inferences}

无法得知(UNKNOWABLE): ${l.unknowable}`).join('\n\n')}

=== ADVERSARIAL VERIFICATION ===
${verdicts.map(v => `CLAIM (${v.claim.from}): ${v.claim.statement}
  -> ${v.refuted ? 'REFUTED' : 'SURVIVED'} (${v.confidence})
  ${v.reasoning}
  corrected: ${v.corrected_statement}`).join('\n\n')}

Structure the answer as:

1. 用户论点的直接裁决 — take the thesis apart into its component claims and rule on each SEPARATELY.
   Lead with the WEAKEST link. Say clearly which links are factually true, which are true but
   already priced, and which are simply broken as a causal chain. A table works well here.

2. 关键量化 — the number that settles the central question, including what is not disclosed and the
   tightest bound derivable. If the transmission from the holder's catalyst to revenue is weak or
   lagged, give the number and the mechanism.

3. 基本面到底怎么样 — separate from the thesis question. What IS growing, what the real multiple is,
   and critically whether any drawdown was estimate cuts or multiple compression. Handle the
   GAAP-versus-adjusted gap and SBC honestly.

4. 真正的风险清单 — ranked, with the factual record for each. Not a generic risk-factor recitation.
   Where a widely-feared risk is NOT currently supported by evidence, say so — do not manufacture
   bearishness for balance.

5. 持有者的决策算术 — sizing, stop and options numbers. Frame it as arithmetic, not advice. Include
   the sunk-cost point about the cost basis.

6. 可验证的观察点 — 3-5 dated, falsifiable items, each with a numeric threshold.

7. 什么会让我改变看法 — the specific observations that would make the opposite case work.

8. 台账登记 — output the exact fund_forward.register(...) call that logs this verdict, with the
   checkpoints from section 6. verdict is supports/mixed/undermines/irrelevant. Use resolver:'auto'
   ONLY for metrics in fund_forward.AUTO_METRICS; anything a filing states in prose (segment splits,
   litigation outcomes, related-party shares) is 'manual'. Marking a manual checkpoint as auto makes
   the ledger look self-maintaining while quietly never resolving.

Be blunt. Attribute every number. Where something is genuinely not knowable from public disclosure,
say so rather than guessing.

End with an explicit statement that this is analysis, not investment advice, and that the position
sizing decision depends on the holder's own portfolio and risk tolerance, which you do not know.`,
  { label: 'verdict', phase: 'Synthesize', effort: 'max' })

return {
  symbol: SYMBOL,
  lines: lines.map(l => ({ key: l.key, impact: l.thesis_impact })),
  claims_attacked: verdicts.length,
  survived: verdicts.filter(v => !v.refuted).map(v => `[${v.claim.from}] ${v.claim.statement}`),
  refuted: verdicts.filter(v => v.refuted).map(v => `[${v.claim.from}] ${v.corrected_statement}`),
  synthesis,
}
