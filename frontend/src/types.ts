export type Method = 0 | 1 | 2;

export interface ApiInput {
  id: number | string;
  n: number;
  alpha: number;
  method: Method;
}

export interface Recommendation {
  appid: number;
  name: string;
  review_score: number;
  score: number;
}
