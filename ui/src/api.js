import { useAuth } from "@clerk/clerk-react";

export const API_BASE = 'http://localhost:8000';

export function useApi() {
  const { getToken } = useAuth();

  const apiFetch = async (url, options = {}) => {
    const token = await getToken();
    const headers = {
      ...options.headers,
      Authorization: `Bearer ${token}`
    };
    return fetch(url, { ...options, headers });
  };

  return { apiFetch };
}