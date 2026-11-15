import type { ApiInput, Recommendation } from './types';

const MOCK_DATA: Recommendation[] = [
  {
    appid: 400,
    name: 'Portal',
    review_score: 9,
    score: 0.782860517501831,
  },
  {
    appid: 2012320,
    name: 'Parallel Experiment',
    review_score: 8,
    score: 0.7740064263343811,
  },
];

export async function fetchDiscovery(params: ApiInput): Promise<Recommendation[]> {
  await new Promise((resolve) => setTimeout(resolve, 300));

  console.log('fetchDiscovery params:', params);

  return MOCK_DATA
}