// Axios client for the FastAPI backend. Adds the login token to every request and turns
// the backend's error envelope {"error": {code, message}} into a readable message.
import axios from 'axios';

const TOKEN_KEY = 'nmt_token';
const USER_KEY = 'nmt_user';

export const api = axios.create({ baseURL: '/api/v1', timeout: 120000 });

export function getSession() {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const username = localStorage.getItem(USER_KEY);
    return token ? { token, username } : null;
  } catch {
    return null;
  }
}

export function saveSession(token, username) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, username);
  } catch { /* storage unavailable: session lasts until reload */ }
  api.defaults.headers.common.Authorization = `Bearer ${token}`;
}

export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  } catch { /* ignore */ }
  delete api.defaults.headers.common.Authorization;
}

const existing = getSession();
if (existing) api.defaults.headers.common.Authorization = `Bearer ${existing.token}`;

let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && !err.config?.url?.startsWith('/auth/')) onUnauthorized();
    return Promise.reject(err);
  },
);

export function errorMessage(err) {
  const e = err?.response?.data?.error;
  if (e) {
    const details = e.details?.map((d) => `${d.field}: ${d.message}`).join('; ');
    return details ? `${e.message} (${details})` : e.message;
  }
  if (err?.code === 'ECONNABORTED') return 'The request timed out.';
  if (!err?.response) return 'Cannot reach the API server. Is the backend running on port 8000?';
  return `Request failed (${err.response.status}).`;
}
