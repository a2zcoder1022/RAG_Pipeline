export async function generateWithOllama(prompt) {
  const response = await fetch(`/api/zero_shot_prompt/ollama?prompt=${encodeURIComponent(prompt)}`)

  let result
  try {
    result = await response.json()
  } catch {
    throw new Error('The API returned an invalid JSON response.')
  }

  if (!response.ok) {
    const detail = typeof result.detail === 'string'
      ? result.detail
      : `The API returned status ${response.status}.`
    throw new Error(detail)
  }

  if (typeof result.response !== 'string') {
    throw new Error('The API response is missing its string "response" field.')
  }

  return result.response
}
