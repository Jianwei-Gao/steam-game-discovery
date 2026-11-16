import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent as ReactMouseEvent, SubmitEvent } from 'react';
import { fetchDiscovery } from './api';
import type { GameListItem, Recommendation, Method } from './types';
import gameListRaw from './game_list.json';
import './App.css';

const gameList = gameListRaw as GameListItem[];
const gameListByAppid = new Map<number, GameListItem>();
for (const g of gameList) gameListByAppid.set(g.appid, g);

const METHOD_OPTIONS: { value: Method; label: string }[] = [
  { value: 0, label: '0' },
  { value: 1, label: '1' },
  { value: 2, label: '2' },
];

const MAX_SUGGESTIONS = 8;

function searchGames(query: string, games: GameListItem[], limit: number): GameListItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const scored: { game: GameListItem; score: number }[] = [];
  for (const game of games) {
    const name = game.name.toLowerCase();
    const appid = String(game.appid);
    let score = 0;
    if (appid === q) score = 100;
    else if (appid.startsWith(q)) score = 80;
    else if (name === q) score = 70;
    else if (name.startsWith(q)) score = 60;
    else if (name.includes(q)) score = 40;
    else if (appid.includes(q)) score = 30;
    if (score > 0) scored.push({ game, score });
  }

  scored.sort(
    (a, b) => b.score - a.score || a.game.name.localeCompare(b.game.name),
  );
  return scored.slice(0, limit).map((s) => s.game);
}

export default function App() {
  const [id, setId] = useState('');
  const [n, setN] = useState('5');
  const [alpha, setAlpha] = useState('0.5');
  const [method, setMethod] = useState<Method>(0);

  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- autocomplete state ---
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  const suggestions = useMemo(
    () => searchGames(id, gameList, MAX_SUGGESTIONS),
    [id],
  );

  useEffect(() => {
    function onDocMouseDown(e: globalThis.MouseEvent) {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, []);

  function selectGame(game: GameListItem) {
    setId(game.name);
    setShowDropdown(false);
  }

  function handleIdKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (!showDropdown || suggestions.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex(
        (i) => (i - 1 + suggestions.length) % suggestions.length,
      );
    } else if (e.key === 'Enter') {
      // Only intercept Enter if a suggestion is actually highlighted
      e.preventDefault();
      selectGame(suggestions[highlightedIndex]);
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    }
  }

  function handleOptionMouseDown(e: ReactMouseEvent<HTMLLIElement>) {
    e.preventDefault();
  }

  async function handleSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setShowDropdown(false);

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
    <div className="layout">
      <aside className="sidebar">
        <h1 className="title">Steam Game Discovery</h1>

        <form onSubmit={handleSubmit} className="form">
          <div className="field">
            <label htmlFor="id">id (game name or appid)</label>
            <div className="autocomplete" ref={autocompleteRef}>
              <input
                id="id"
                type="text"
                value={id}
                autoComplete="off"
                onChange={(e) => {
                  setId(e.target.value);
                  setHighlightedIndex(0);
                  setShowDropdown(true);
                }}
                onFocus={() => {
                  if (suggestions.length > 0) setShowDropdown(true);
                }}
                onKeyDown={handleIdKeyDown}
                placeholder="e.g. Portal or 400"
                required
              />

              {showDropdown && suggestions.length > 0 && (
                <ul className="dropdown" role="listbox">
                  {suggestions.map((game, idx) => (
                    <li
                      key={game.appid}
                      role="option"
                      aria-selected={idx === highlightedIndex}
                      className={`dropdown-option${
                        idx === highlightedIndex ? ' highlighted' : ''
                      }`}
                      onMouseDown={handleOptionMouseDown}
                      onMouseEnter={() => setHighlightedIndex(idx)}
                      onClick={() => selectGame(game)}
                    >
                      <div className="dropdown-line">
                        <span className="dropdown-name">{game.name}</span>
                        <span className="dropdown-appid">{game.appid}</span>
                      </div>
                      <div className="dropdown-desc">
                        {game.short_description}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="field">
            <label htmlFor="n">n (integer)</label>
            <input
              id="n"
              type="number"
              step="1"
              value={n}
              onChange={(e) => setN(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="alpha">alpha (number)</label>
            <input
              id="alpha"
              type="number"
              step="any"
              value={alpha}
              onChange={(e) => setAlpha(e.target.value)}
            />
          </div>

          <fieldset className="field">
            <legend>method</legend>
            <div className="radio-group">
              {METHOD_OPTIONS.map((opt) => (
                <label key={opt.value} className="radio">
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
            </div>
          </fieldset>

          <button type="submit" disabled={loading}>
            {loading ? 'Loading…' : 'Submit'}
          </button>

          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
        </form>
      </aside>

      <main className="content">
        {results.length === 0 && !loading && !error && (
          <p className="empty">No results yet — submit a query to begin.</p>
        )}

        <ul className="results">
          {results.map((game) => {
            const info = gameListByAppid.get(game.appid);
            return (
              <li key={game.appid} className="result">
                <div className="result-info">
                  <a
                    href={`https://store.steampowered.com/app/${game.appid}/`}
                    target="_blank"
                    rel="noreferrer"
                    className="result-name"
                  >
                    {game.name}
                  </a>
                  <span className="result-appid">{game.appid}</span>
                  {info?.short_description && (
                    <p className="result-desc">{info.short_description}</p>
                  )}
                </div>
                <img
                  className="result-image"
                  src={`https://steamcdn-a.akamaihd.net/steam/apps/${game.appid}/header.jpg`}
                  alt={game.name}
                  width={616}
                  height={353}
                  loading="lazy"
                />
              </li>
            );
          })}
        </ul>
      </main>
    </div>
  );
}