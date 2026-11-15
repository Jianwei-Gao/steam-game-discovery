import { useState } from 'react';
import type { SubmitEvent } from 'react';
import { fetchDiscovery } from './api';
import type { Recommendation, Method } from './types';

const METHODS: { value: Method; label: string }[] = [
  { value: 0, label: '0' },
  { value: 1, label: '1' },
  { value: 2, label: '2' },
];

export default function App() {
  const [id, setId] = useState<string>('');
  const [n, setN] = useState('5');
  const [alpha, setAlpha] = useState('0.5');
  const [method, setMethod] = useState<Method>(0);

  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    // --- validation ---
    const trimmedId = id.trim();
    if (!trimmedId) {
      setError('id is required.');
      return;
    }

    const nNum = Number(n);
    if (!Number.isInteger(nNum) || nNum <= 0) {
      setError('n must be a positive integer.');
      return;
    }

    const alphaNum = Number(alpha);
    if (!Number.isFinite(alphaNum)) {
      setError('alpha must be a number.');
      return;
    }

    // id accepts integer or string: convert only if it looks like an integer
    const parsedId: number | string = /^-?\d+$/.test(trimmedId)
      ? Number(trimmedId)
      : trimmedId;

    setLoading(true);
    try {
      const data = await fetchDiscovery({
        id: parsedId,
        n: nNum,
        alpha: alphaNum,
        method,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Steam Game Discovery</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="id">id</label>
          <input
            id="id"
            type="text"
            value={id}
            onChange={(e) => setId(e.target.value)}
            placeholder="e.g. 620 or portal 2"
            required
          />
        </div>

        <div>
          <label htmlFor="n">n</label>
          <input
            id="n"
            type="number"
            step="1"
            value={n}
            onChange={(e) => setN(e.target.value)}
          />
        </div>

        <div>
          <label htmlFor="alpha">alpha</label>
          <input
            id="alpha"
            type="number"
            step="any"
            value={alpha}
            onChange={(e) => setAlpha(e.target.value)}
          />
        </div>

        <fieldset>
          <legend>method</legend>
          {METHODS.map((opt) => (
            <label key={opt.value}>
              <input
                type="radio"
                name="method"
                value={opt.value}
                checked={method === opt.value}
                onChange={() => setMethod(opt.value)}
              />
              {opt.label}
            </label>
          ))}
        </fieldset>

        <button type="submit" disabled={loading}>
          {loading ? 'Loading…' : 'Submit'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}

      <ul>
        {results.map((game) => (
          <li key={game.appid}>
            <a
              href={`https://store.steampowered.com/app/${game.appid}/`}
              target="_blank"
              rel="noreferrer"
            >
              {game.name}
            </a>{' '}
            <span>{game.appid}</span>
            <br />
            <img
              src={`https://shared.steamstatic.com/store_item_assets/steam/apps/${game.appid}/capsule_616x353.jpg`}
              alt={game.name}
              width={616}
              height={353}
            />
          </li>
        ))}
      </ul>
    </main>
  );
}