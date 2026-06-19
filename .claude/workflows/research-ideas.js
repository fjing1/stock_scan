export const meta = {
  name: 'research-ideas',
  description: 'Fetch external stock-market strategy research, cross-check against this repo\'s prior studies, score each idea\'s maturity with adversarial verification, and return a ranked shortlist for a human go/no-go decision.',
  whenToUse: 'When you want to scan the world of trading-strategy research for mature, implementable, NEW-to-this-repo edges (entry/exit/ranking rules) and get a ranked shortlist to choose from.',
  phases: [
    { title: 'Fetch', detail: 'parallel web research across 6 strategy lenses + read this repo\'s prior studies' },
    { title: 'Cross-check', detail: 'dedup candidates and label NEW vs already-tested vs dead-end here' },
    { title: 'Assess', detail: 'maturity-score each shortlisted idea, then adversarially verify with two skeptics' },
  ],
}

const TODAY = (args && args.date) || 'today'
const REPO = '/Users/feijing/github.com/stock_scan'

// ---- Research lenses: distinct angles so the fetchers don't overlap ----
const LENSES = [
  { key: 'academic-anomalies', focus: 'Peer-reviewed and SSRN/arXiv working-paper research on equity cross-sectional anomalies and factors, prioritizing post-2015 work plus replication and decay studies (e.g. factor-zoo replication, "Does Academic Research Destroy Stock Return Predictability", out-of-sample factor performance, McLean-Pontiff style decay).' },
  { key: 'quant-practitioner-blogs', focus: 'Reputable quant practitioner sources (Alpha Architect, Quantocracy-aggregated blogs, Robot Wealth, QuantStart, Newfound/ReSolve, practitioner SSRN notes) on robust, implementable systematic equity strategies, emphasizing out-of-sample robustness over backtest hype.' },
  { key: 'momentum-trend', focus: 'Cross-sectional and time-series momentum, dual momentum, relative-strength sector/ETF rotation, and trend-following on equities/ETFs - including how to avoid momentum crashes and the role of trend/volatility filters.' },
  { key: 'mean-reversion-short-term', focus: 'Short-term mean-reversion and swing edges on equities: RSI(2)/Connors-style, internal-bar strength, oversold-in-uptrend pullbacks, gaps, and overnight/seasonality effects. This repo already found a dip-in-uptrend edge - find adjacent or stronger variants.' },
  { key: 'regime-vol-risk', focus: 'Regime filters, volatility targeting/scaling, trend-based market-timing overlays, and crash-protection rules that improve the risk-adjusted returns of an equity swing system (e.g. moving-average regime gates, credit/breadth filters, vol-control position sizing).' },
  { key: 'crosssectional-factor-combos', focus: 'Market-neutral / cross-sectional alpha construction: combining value/quality/momentum/low-vol factors, factor timing, turnover reduction (rebalance buffers), and ensemble/stacking methods for long-short equity.' },
]

// ---------------------------- Schemas ----------------------------
const CANDIDATES_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          category: { type: 'string' },
          thesis: { type: 'string' },
          mechanism: { type: 'string' },
          evidence: { type: 'string' },
          claimedEdge: { type: 'string' },
          dataNeeded: { type: 'string' },
          sources: { type: 'array', items: { type: 'string' } },
          maturitySignals: { type: 'string' },
        },
        required: ['name', 'category', 'thesis', 'mechanism', 'evidence', 'dataNeeded', 'sources'],
      },
    },
  },
  required: ['lens', 'candidates'],
}

const LEDGER_SCHEMA = {
  type: 'object',
  properties: {
    triedIdeas: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          status: { type: 'string', enum: ['edge-found', 'no-edge', 'killed'] },
          note: { type: 'string' },
        },
        required: ['name', 'status', 'note'],
      },
    },
  },
  required: ['triedIdeas'],
}

const CROSSCHECK_SCHEMA = {
  type: 'object',
  properties: {
    ideas: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          category: { type: 'string' },
          thesis: { type: 'string' },
          evidence: { type: 'string' },
          dataNeeded: { type: 'string' },
          sources: { type: 'array', items: { type: 'string' } },
          repoStatus: { type: 'string', enum: ['NEW', 'OVERLAPS-TESTED', 'DEAD-END'] },
          repoNote: { type: 'string' },
          shortlist: { type: 'boolean' },
        },
        required: ['name', 'category', 'thesis', 'repoStatus', 'shortlist'],
      },
    },
  },
  required: ['ideas'],
}

const MATURITY_SCHEMA = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    scores: {
      type: 'object',
      properties: {
        evidence: { type: 'number' },
        oosReplication: { type: 'number' },
        implementability: { type: 'number' },
        capacity: { type: 'number' },
        robustness: { type: 'number' },
      },
      required: ['evidence', 'oosReplication', 'implementability', 'capacity', 'robustness'],
    },
    totalScore: { type: 'number' },
    rationale: { type: 'string' },
    implementationSketch: { type: 'string' },
    risks: { type: 'string' },
  },
  required: ['scores', 'totalScore', 'rationale', 'implementationSketch', 'risks'],
}

const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: {
    strongestObjection: { type: 'string' },
    likelyDecayed: { type: 'boolean' },
    overfitRisk: { type: 'string', enum: ['low', 'medium', 'high'] },
    verdict: { type: 'string', enum: ['promising', 'marginal', 'reject'] },
    reason: { type: 'string' },
  },
  required: ['strongestObjection', 'overfitRisk', 'verdict', 'reason'],
}

// ---------------------------- Phase 1: Fetch ----------------------------
phase('Fetch')

const webPrompt = (lens) => `You are a quantitative-research scout finding tradable strategy ideas. Research lens: ${lens.focus}

FIRST load web tools: call ToolSearch with query "select:WebSearch,WebFetch", then use WebSearch and WebFetch. Do 3-6 searches and fetch the 3-6 most credible sources (papers, replication studies, reputable practitioner writeups). Favor ideas with independent, out-of-sample evidence over backtest hype.

For each DISTINCT idea (aim for 4-8), capture: a short name; category; thesis (the edge in 1-3 sentences); mechanism (the economic reason it should work); evidence (papers/replications/live records - be specific and cite what you actually read); claimedEdge (magnitude/horizon if stated); dataNeeded (price/volume/fundamentals/options/macro); sources (URLs you actually read); maturitySignals (independent replications, years live, known decay).

Scope: only ideas a US-equity/ETF SWING system on daily/4H bars could implement. Skip intraday-HFT, options-heavy, or alt-data-only ideas. Return structured data only - your final output IS the data object.`

const repoLedgerPrompt = `Catalog this repo's PRIOR research so we can separate new ideas from already-tested ones.

Read ${REPO}/tradingview_scripts/RESEARCH.md in full, and list the .py files in ${REPO}/tradingview_scripts/ (read a few of the less-obvious ones if their purpose is unclear from the filename).

For every strategy/signal idea this repo has already studied, record: name; status = "edge-found" / "no-edge" / "killed"; and a one-line note of what was tested and concluded. RESEARCH.md is the source of truth for findings (e.g. dip-in-uptrend mean-reversion and a market-neutral factor ensemble FOUND edge; MTF turn-alignment and naive sector/macro timing were dead ends). Return structured data only.`

const [candidateResults, ledger] = await Promise.all([
  parallel(LENSES.map((lens) => () =>
    agent(webPrompt(lens), { label: `fetch:${lens.key}`, phase: 'Fetch', schema: CANDIDATES_SCHEMA }))),
  agent(repoLedgerPrompt, { label: 'repo-ledger', phase: 'Fetch', schema: LEDGER_SCHEMA, agentType: 'Explore' }),
])

const goodLenses = candidateResults.filter(Boolean)
const allCandidates = goodLenses.flatMap((r) => (r.candidates || []).map((c) => ({ ...c, lens: r.lens })))
const triedIdeas = (ledger && ledger.triedIdeas) || []
log(`Fetched ${allCandidates.length} raw candidate ideas across ${goodLenses.length}/${LENSES.length} lenses; repo ledger has ${triedIdeas.length} prior studies.`)

// ---------------------------- Phase 2: Cross-check ----------------------------
phase('Cross-check')

const crosscheck = await agent(`You are deduplicating and triaging trading-strategy candidates against a repo's prior research.

CANDIDATE IDEAS (JSON):
${JSON.stringify(allCandidates)}

THIS REPO HAS ALREADY TESTED (JSON):
${JSON.stringify(triedIdeas)}

Tasks:
1. Merge near-duplicate candidate ideas across lenses into single entries (union their sources, keep the best thesis/evidence/dataNeeded).
2. For each unique idea set repoStatus: "NEW" (not tested here), "OVERLAPS-TESTED" (close to something tried - note which in repoNote), or "DEAD-END" (essentially something this repo already found has no edge / killed, e.g. MTF turn-alignment or naive sector/macro timing).
3. Set shortlist=true for the most promising ideas to assess further - prioritize strong independent/OOS evidence AND novelty for this repo. Set shortlist=false for DEAD-ENDs and thin/hype ideas. Shortlist AT MOST 12.

Return the FULL deduped list (every unique idea) with these fields.`, { label: 'crosscheck', phase: 'Cross-check', schema: CROSSCHECK_SCHEMA })

const unique = (crosscheck && crosscheck.ideas) || []
const shortlist = unique.filter((i) => i.shortlist && i.repoStatus !== 'DEAD-END').slice(0, 12)
log(`${unique.length} unique ideas after dedup; ${shortlist.length} shortlisted for maturity assessment.`)

// ---------------------------- Phase 3: Assess (maturity -> adversarial verify) ----------------------------
phase('Assess')

const assessed = await pipeline(
  shortlist,
  (idea) => agent(`Score the MATURITY and implementability of this trading-strategy idea for a US-equity/ETF SWING system on daily/4H bars. Available data: yfinance prices/volume/fundamentals + FRED macro (keyless) + a ~1,243-name universe. Evaluation house style: train/test split at 2019-01-01, detrended-vs-SPY metrics, "no edge = kill it".

Assess ONLY from the evidence in the IDEA JSON plus your own knowledge. Do NOT use web search or fetch tools - the research has already been gathered; reason over it. Be concise.

IDEA (JSON): ${JSON.stringify(idea)}

Score each 0-5 (5=best): evidence (strength + independence of supporting research); oosReplication (independent / out-of-sample replication exists); implementability (buildable with the available data + this pipeline); capacity (turnover/holding-period friendly for swing trading, not HFT); robustness (low overfitting/decay risk, simple, economically grounded). totalScore = sum (0-25).

Provide rationale, implementationSketch (which data + how it would slot into a standalone Python study like the repo's existing ones, respecting the 2019-01-01 OOS split and detrended evaluation), and key risks. Return structured data only.`, { label: `maturity:${(idea.name || 'idea').slice(0, 22)}`, phase: 'Assess', schema: MATURITY_SCHEMA }),
  (maturity, idea) => parallel([
    () => agent(`Adversarially evaluate whether this trading edge is REAL and still tradable, or likely overfit / decayed / arbitraged-away. Be a skeptic; default to doubt when evidence is thin or in-sample only. Reason from the evidence provided and your own knowledge - do NOT use web search/fetch tools. Be concise.
IDEA: ${JSON.stringify(idea)}
MATURITY ASSESSMENT: ${JSON.stringify(maturity)}
Give the strongest objection, whether it is likelyDecayed, overfitRisk (low/medium/high), and a verdict of promising/marginal/reject with a reason.`, { label: `skeptic-A:${(idea.name || 'idea').slice(0, 18)}`, phase: 'Assess', schema: SKEPTIC_SCHEMA }),
    () => agent(`As an IMPLEMENTATION skeptic, judge whether THIS repo could actually build and benefit from this idea given only yfinance + FRED data on daily/4H bars, and whether it overlaps too much with edges the repo already has (dip-in-uptrend mean-reversion; market-neutral factor ensemble). Reject if redundant or undeliverable with the available data. Reason from the evidence provided and your own knowledge - do NOT use web search/fetch tools. Be concise.
IDEA: ${JSON.stringify(idea)}
Give the strongest objection, likelyDecayed, overfitRisk (low/medium/high), and verdict promising/marginal/reject with a reason.`, { label: `skeptic-B:${(idea.name || 'idea').slice(0, 18)}`, phase: 'Assess', schema: SKEPTIC_SCHEMA }),
  ]).then((votes) => {
    const v = votes.filter(Boolean)
    const rejects = v.filter((x) => x.verdict === 'reject').length
    const promisings = v.filter((x) => x.verdict === 'promising').length
    const consensus = rejects >= 1 && rejects >= promisings ? 'reject' : (promisings >= 1 ? 'promising' : 'marginal')
    return { ...idea, maturity, skeptics: v, consensus }
  }),
)

const final = assessed.filter(Boolean)
final.sort((a, b) => ((b.maturity && b.maturity.totalScore) || 0) - ((a.maturity && a.maturity.totalScore) || 0))

return {
  date: TODAY,
  rawCount: allCandidates.length,
  uniqueCount: unique.length,
  shortlistCount: shortlist.length,
  deadEnds: unique.filter((i) => i.repoStatus === 'DEAD-END').map((i) => ({ name: i.name, repoNote: i.repoNote })),
  ideas: final,
}
