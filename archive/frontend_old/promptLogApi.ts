import type { PromptTransaction, CostDashboardData } from '../types/promptLog';

const mockTransactions: PromptTransaction[] = [
  {
    id: '13910cd4-b3c3-4fbf-aa16-0674cacfb172',
    provider: 'GENERIC_OPENAI',
    model: 'google/gemma-4-26b-a4b-qat',
    feature: 'aa_agent_planner',
    method: 'generate_json',
    timestamp: '2026-09-24T09:40:27Z',
    inputTokens: 7259,
    outputTokens: 259,
    totalTokens: 7518,
    cost: 0.0000,
    latencyMs: 2785,
    status: 'success',
    prompt: 'User question: tạo 1 dashboard để quản lý vùng có thể xem performance của họ',
    systemInstruction: 'You are the planning step of a Data Agent called "Ecommerce Agent". Produce a JSON plan for answering the user\'s question. Do NOT attempt to answer, only plan.',
    response: '{"intent": "Người dùng muốn xây dựng một bảng điều khiển để quản lý các vùng để theo dõi hiệu suất.", "granularity": {"dimension": "breakdown"}}'
  },
  {
    id: '13910cd4-b3c3-4fbf-aa16-0674cacfb173',
    provider: 'OPENAI_COMPATIBLE',
    model: 'text-embedding-qwen3-embedding-8b',
    feature: 'embedding',
    method: 'embedding',
    timestamp: '2026-09-24T09:40:28Z',
    inputTokens: 15,
    outputTokens: 0,
    totalTokens: 15,
    cost: 0.0000,
    latencyMs: 516,
    status: 'success',
    prompt: 'Evaluate image context...',
    systemInstruction: 'Generate embedding for comparison.',
    response: '[0.012, 0.034, -0.012, ...]'
  },
  {
    id: '13910cd4-b3c3-4fbf-aa16-0674cacfb174',
    provider: 'GEMINI',
    model: 'gemini-3.1-flash-lite',
    feature: 'aa_session_title',
    method: 'generate',
    timestamp: '2026-09-24T09:38:07Z',
    inputTokens: 613,
    outputTokens: 8,
    totalTokens: 621,
    cost: 0.000165,
    latencyMs: 1378,
    status: 'success',
    prompt: 'What is the breed of this dog?',
    systemInstruction: 'You are a classification assistant.',
    response: 'Golden Retriever'
  }
];

export const promptLogApi = {
  listTransactions: async (): Promise<PromptTransaction[]> => {
    return new Promise(resolve => setTimeout(() => resolve(mockTransactions), 500));
  },
  
  getDashboardData: async (): Promise<CostDashboardData> => {
    return new Promise(resolve => setTimeout(() => resolve({
      totalCost: 0.000165,
      totalTransactions: 760,
      avgCostPerTransaction: 0.000165,
      totalTokens: 8900000,
      cacheRate: 0,
      cacheSavings: 0,
      costTrend: [
        { date: '2026-09-21', cost: 0.000010 },
        { date: '2026-09-22', cost: 0.000015 },
        { date: '2026-09-23', cost: 0.000020 },
        { date: '2026-09-24', cost: 0.000165 }
      ],
      costByFeature: [
        { feature: 'aa_session_title', cost: 0.000165 },
        { feature: 'aa_agent_planner', cost: 0 },
        { feature: 'embedding', cost: 0 }
      ],
      costByProvider: [
        { provider: 'GEMINI', cost: 0.000165 },
        { provider: 'GENERIC_OPENAI', cost: 0 },
        { provider: 'OPENAI_COMPATIBLE', cost: 0 }
      ],
      costByModel: [
        { model: 'gemini-3.1-flash-lite', cost: 0.000165 },
        { model: 'google/gemma-4-26b', cost: 0 },
        { model: 'text-embedding-qwen', cost: 0 }
      ]
    }), 500));
  }
};
