import type { ApiInput, Recommendation } from './types';

const API_LINK = 'http://localhost:8000'

export async function fetchDiscovery(params: ApiInput): Promise<Recommendation[]> {
  const qs = new URLSearchParams({
    id: String(params.id),
    n: String(params.n),
    alpha: String(params.alpha),
    method: String(params.method),
  });

  let res: Response;
  try {
    res = await fetch(`${API_LINK}/query?${qs}`, {
      method: 'GET',
      headers: { accept: '*/*' },
    });
  } 
  catch {
    throw new Error('Error: could not reach the API.');
  }

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body: unknown = await res.json();
      if (
        body 
        && typeof body === 'object' 
        && 'detail' in body 
        && typeof (body as { detail: unknown }).detail === 'string'
      ) {
        detail = (body as { detail: string }).detail;
      }
    } 
    catch {
      throw new Error('Unexpected Error: bad return from api.', {cause: detail});
    }
    throw new Error(detail);
  }

  const data: unknown = await res.json();
  if (!Array.isArray(data)) {
    throw new Error('Unexpected response shape from API.');
  }
  return data as Recommendation[];
}