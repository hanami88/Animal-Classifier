export interface PromptTransaction {
  id: string;
  provider: string;
  model: string;
  feature: string;
  method: string;
  timestamp: string;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  latencyMs: number;
  status: 'success' | 'error' | 'pending';
  prompt: string;
  systemInstruction: string;
  response: string;
}

export interface CostDashboardData {
  totalCost: number;
  totalTransactions: number;
  avgCostPerTransaction: number;
  totalTokens: number;
  cacheRate: number;
  cacheSavings: number;
  costTrend: { date: string; cost: number }[];
  costByFeature: { feature: string; cost: number }[];
  costByProvider: { provider: string; cost: number }[];
  costByModel: { model: string; cost: number }[];
}
