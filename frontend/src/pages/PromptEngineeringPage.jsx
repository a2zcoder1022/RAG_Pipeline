import { useMemo, useState } from 'react'

const initialExamples = [
  {
    input: 'I was charged twice for this month’s subscription.',
    expected_output: 'Billing',
  },
  {
    input: 'The app closes whenever I open a report.',
    expected_output: 'Technical Support',
  },
  {
    input: 'I cannot sign in because my password reset link expired.',
    expected_output: 'Account Access',
  },
]

const defaultInstruction =
  'Classify the support message as exactly one of: Billing, Technical Support, or Account Access. Return only the category name.'

function PromptEngineeringPage() {
  const [mode, setMode] = useState('zero_shot')
  const [model, setModel] = useState('llama3.2')
  const [instruction, setInstruction] = useState(defaultInstruction)
  const [examples, setExamples] = useState(initialExamples)
  const [input, setInput] = useState(
    'I changed phones and can no longer receive the sign-in verification code.',
  )
  const [expectedOutput, setExpectedOutput] = useState('Account Access')
  const [temperature, setTemperature] = useState('0')
  const [topP, setTopP] = useState('0.9')
  const [response, setResponse] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const promptPreview = useMemo(() => {
    const sections = [instruction.trim()]
    if (mode === 'few_shot') {
      examples.forEach((example, index) => {
        if (example.input.trim() && example.expected_output.trim()) {
          sections.push(
            `Example ${index + 1}\nInput: ${example.input.trim()}\nExpected output: ${example.expected_output.trim()}`,
          )
        }
      })
    }
    sections.push(`Input: ${input.trim()}\nOutput:`)
    return sections.filter(Boolean).join('\n\n')
  }, [examples, input, instruction, mode])

  const isCorrect =
    response.trim() &&
    expectedOutput.trim() &&
    response.trim().toLowerCase() === expectedOutput.trim().toLowerCase()

  const updateExample = (index, field, value) => {
    setExamples((current) => current.map((example, exampleIndex) => (
      exampleIndex === index ? { ...example, [field]: value } : example
    )))
  }

  const runPrompt = async (event) => {
    event.preventDefault()
    setResponse('')
    setError('')

    if (!instruction.trim() || !input.trim() || !model.trim()) {
      setError('Enter an instruction, a message to classify, and a model name.')
      return
    }

    const temperatureValue = Number(temperature)
    const topPValue = Number(topP)
    if (
      !Number.isFinite(temperatureValue)
      || temperatureValue < 0
      || temperatureValue > 2
      || !Number.isFinite(topPValue)
      || topPValue <= 0
      || topPValue > 1
    ) {
      setError('Temperature must be from 0 to 2, and top-p must be greater than 0 and at most 1.')
      return
    }

    const selectedExamples = mode === 'few_shot'
      ? examples.filter((example) => example.input.trim() && example.expected_output.trim())
      : []

    if (mode === 'few_shot' && selectedExamples.length === 0) {
      setError('Add at least one complete example for few-shot prompting.')
      return
    }

    setIsSubmitting(true)
    try {
      const apiResponse = await fetch('/api/prompt-engineering/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          model: model.trim(),
          instruction: instruction.trim(),
          input: input.trim(),
          examples: selectedExamples,
          temperature: temperatureValue,
          top_p: topPValue,
        }),
      })

      const responseText = await apiResponse.text()
      let result
      try {
        result = JSON.parse(responseText)
      } catch {
        if (!apiResponse.ok) {
          throw new Error(`The API returned status ${apiResponse.status}. Check that your FastAPI prompt endpoint is running.`)
        }
        throw new Error('The API response was not valid JSON.')
      }

      if (!apiResponse.ok) {
        const detail = typeof result.detail === 'string'
          ? result.detail
          : `The API returned status ${apiResponse.status}.`
        throw new Error(detail)
      }

      if (typeof result.response !== 'string') {
        throw new Error('Expected the API to return a string in its "response" field.')
      }

      setResponse(result.response)
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? 'Could not reach the API. Check that FastAPI is running and has the prompt-engineering endpoint.'
          : requestError.message,
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="page-content prompt-lab-page">
      <section className="prompt-lab-heading">
        <div>
          <div className="section-kicker">PROMPT ENGINEERING</div>
          <h1>Prompt Lab</h1>
          <p>Try instructions and examples, then see how Ollama responds.</p>
        </div>
        <span className="prompt-lab-mark" aria-hidden="true">⌘</span>
      </section>

      <div className="prompt-lab-grid">
        <form className="prompt-form-panel" onSubmit={runPrompt}>
          <div className="form-section">
            <div className="form-section-title">
              <span className="form-step">01</span>
              <div><strong>Choose a prompting style</strong><small>Examples are optional in zero-shot mode.</small></div>
            </div>
            <div className="mode-toggle" role="group" aria-label="Prompting style">
              <button
                className={mode === 'zero_shot' ? 'selected' : ''}
                onClick={() => setMode('zero_shot')}
                type="button"
              >
                Zero-shot <span>No examples</span>
              </button>
              <button
                className={mode === 'few_shot' ? 'selected' : ''}
                onClick={() => setMode('few_shot')}
                type="button"
              >
                Few-shot <span>{examples.length} examples</span>
              </button>
            </div>
          </div>

          <div className="form-section">
            <label className="form-section-title" htmlFor="prompt-instruction">
              <span className="form-step">02</span>
              <span><strong>Your instruction</strong><small>Tell the model what task to do and how to respond.</small></span>
            </label>
            <textarea
              id="prompt-instruction"
              onChange={(event) => setInstruction(event.target.value)}
              rows={3}
              value={instruction}
            />
          </div>

          {mode === 'few_shot' && (
            <div className="form-section examples-section">
              <div className="form-section-title">
                <span className="form-step">03</span>
                <div><strong>Example input and answer</strong><small>Examples show the model the pattern you want.</small></div>
              </div>
              <div className="example-editor-list">
                {examples.map((example, index) => (
                  <div className="example-editor" key={index}>
                    <div className="example-editor-heading">
                      <span>Example {index + 1}</span>
                      {examples.length > 1 && (
                        <button
                          aria-label={`Remove example ${index + 1}`}
                          onClick={() => setExamples((current) => current.filter((_, i) => i !== index))}
                          type="button"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                    <label>
                      <span>Input</span>
                      <textarea
                        onChange={(event) => updateExample(index, 'input', event.target.value)}
                        rows={2}
                        value={example.input}
                      />
                    </label>
                    <label>
                      <span>Expected answer</span>
                      <input
                        onChange={(event) => updateExample(index, 'expected_output', event.target.value)}
                        value={example.expected_output}
                      />
                    </label>
                  </div>
                ))}
              </div>
              <button
                className="add-example-button"
                onClick={() => setExamples((current) => [...current, { input: '', expected_output: '' }])}
                type="button"
              >
                + Add an example
              </button>
            </div>
          )}

          <div className="form-section">
            <label className="form-section-title" htmlFor="prompt-input">
              <span className="form-step">{mode === 'few_shot' ? '04' : '03'}</span>
              <span><strong>Try an input</strong><small>This is the new message the model will classify.</small></span>
            </label>
            <textarea
              id="prompt-input"
              onChange={(event) => setInput(event.target.value)}
              rows={3}
              value={input}
            />
            <label className="expected-answer-field">
              <span>Expected answer <small>for your comparison only; not sent to the model</small></span>
              <input
                onChange={(event) => setExpectedOutput(event.target.value)}
                placeholder="e.g. Account Access"
                value={expectedOutput}
              />
            </label>
          </div>

          <details className="generation-settings">
            <summary>Model and generation settings</summary>
            <div className="settings-fields">
              <label>
                Model name
                <input onChange={(event) => setModel(event.target.value)} value={model} />
              </label>
              <label>
                Temperature
                <input
                  max="2"
                  min="0"
                  onChange={(event) => setTemperature(event.target.value)}
                  step="0.1"
                  type="number"
                  value={temperature}
                />
              </label>
              <label>
                Top-p
                <input
                  max="1"
                  min="0"
                  onChange={(event) => setTopP(event.target.value)}
                  step="0.05"
                  type="number"
                  value={topP}
                />
              </label>
            </div>
          </details>

          <button className="primary-button run-prompt-button" disabled={isSubmitting} type="submit">
            {isSubmitting ? 'Waiting for Ollama…' : 'Run prompt'}
            {!isSubmitting && <span aria-hidden="true">↗</span>}
          </button>
          <p className="api-note">This page sends a request to your FastAPI prompt endpoint. The endpoint still needs to be added to the Python backend.</p>
        </form>

        <aside className="prompt-results-column">
          <section className="prompt-preview-panel">
            <div className="panel-heading">
              <div><span className="section-kicker">WHAT THE MODEL SEES</span><h2>Prompt preview</h2></div>
              <span className={`preview-mode ${mode}`}>{mode === 'zero_shot' ? 'Zero-shot' : 'Few-shot'}</span>
            </div>
            <pre>{promptPreview || 'Your prompt preview will appear here.'}</pre>
          </section>

          <section className={`prompt-output-panel ${error ? 'has-error' : ''}`}>
            <div className="panel-heading">
              <div><span className="section-kicker">MODEL RESPONSE</span><h2>Output</h2></div>
              <span className="ollama-status"><span /> Ollama</span>
            </div>
            {error ? (
              <div className="prompt-error" role="alert">{error}</div>
            ) : response ? (
              <>
                <div className="response-text">{response}</div>
                {expectedOutput.trim() && (
                  <div className={`comparison-result ${isCorrect ? 'correct' : 'incorrect'}`}>
                    <span>{isCorrect ? '✓' : '!'}</span>
                    {isCorrect ? 'Matches the expected answer' : `Expected: ${expectedOutput}`}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-response">
                <span className="response-spark">✦</span>
                <strong>Your result will appear here</strong>
                <p>Run a prompt to compare the model’s answer with your expected result.</p>
              </div>
            )}
          </section>
        </aside>
      </div>
    </div>
  )
}

export default PromptEngineeringPage
