import { useMemo, useState } from 'react'
import { generateWithOllama } from '../api/ollama.js'
import dataset from '../../../LLM/PromptEngineering/support_ticket_dataset.json'
import './PromptLabPage.css'

const categories = Object.keys(dataset.labels)

function getPredictedCategory(response) {
  const normalized = response
    .trim()
    .replace(/^[`"']+|[`"'.,!?;:\s]+$/g, '')
    .toLowerCase()

  return categories.find((category) => category.toLowerCase() === normalized) ?? null
}

function createPrompt(example, mode) {
  const sections = [dataset.task_instruction]

  if (mode === 'few_shot') {
    dataset.few_shot_examples.forEach((trainingExample, index) => {
      sections.push(
        `Example ${index + 1}\nInput: ${trainingExample.input}\nExpected output: ${trainingExample.expected_output}`,
      )
    })
  }

  sections.push(`Input: ${example.input}\nOutput:`)
  return sections.join('\n\n')
}

function PromptLabPage() {
  const [mode, setMode] = useState('zero_shot')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [results, setResults] = useState([])
  const [completed, setCompleted] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState('')

  const evaluationExamples = useMemo(
    () => categoryFilter === 'all'
      ? dataset.evaluation_examples
      : dataset.evaluation_examples.filter((example) => example.expected_output === categoryFilter),
    [categoryFilter],
  )

  const categoryMetrics = useMemo(
    () => categories.map((category) => {
      const categoryResults = results.filter((result) => result.expected === category)
      const correct = categoryResults.filter((result) => result.predicted === category).length

      return {
        category,
        correct,
        total: categoryResults.length,
      }
    }),
    [results],
  )

  const correctCount = results.filter((result) => result.correct).length
  const validCount = results.filter((result) => result.predicted !== null).length

  const runEvaluation = async () => {
    setResults([])
    setCompleted(0)
    setError('')
    setIsRunning(true)

    try {
      for (const [index, example] of evaluationExamples.entries()) {
        const prompt = createPrompt(example, mode)
        const rawResponse = await generateWithOllama(prompt)
        const predicted = getPredictedCategory(rawResponse)
        setResults((current) => [...current, {
          id: example.id,
          input: example.input,
          expected: example.expected_output,
          rawResponse,
          predicted,
          correct: predicted === example.expected_output,
        }])
        setCompleted(index + 1)
      }
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? 'Could not reach the API. Start FastAPI and Ollama, then try again.'
          : requestError.message,
      )
    } finally {
      setIsRunning(false)
    }
  }

  const resetResults = () => {
    setResults([])
    setCompleted(0)
    setError('')
  }

  return (
    <div className="page-content prompt-lab-evaluation-page">
      <section className="prompt-lab-heading">
        <div>
          <div className="section-kicker">PROMPT ENGINEERING</div>
          <h1>Prompt Lab</h1>
          <p>Test zero-shot and few-shot prompts against support tickets from your dataset.</p>
        </div>
        <span className="prompt-lab-mark" aria-hidden="true">◎</span>
      </section>

      <section className="dataset-setup-panel">
        <div className="dataset-setup-copy">
          <span className="section-kicker">YOUR TEST SET</span>
          <h2>{dataset.dataset_name.replaceAll('_', ' ')}</h2>
          <p>
            The page uses the dataset’s few-shot examples as prompt context and scores
            only the separate evaluation examples. Evaluation labels stay out of the prompt.
          </p>
          <div className="dataset-counts">
            <span><strong>{dataset.few_shot_examples.length}</strong> training examples</span>
            <span><strong>{dataset.evaluation_examples.length}</strong> held-out examples</span>
            <span><strong>{categories.join(' · ')}</strong></span>
          </div>
        </div>

        <div className="dataset-controls">
          <label>
            Prompt style
            <select
              disabled={isRunning}
              onChange={(event) => {
                setMode(event.target.value)
                resetResults()
              }}
              value={mode}
            >
              <option value="zero_shot">Zero-shot</option>
              <option value="few_shot">Few-shot</option>
            </select>
          </label>
          <label>
            Test category
            <select
              disabled={isRunning}
              onChange={(event) => {
                setCategoryFilter(event.target.value)
                resetResults()
              }}
              value={categoryFilter}
            >
              <option value="all">All categories</option>
              {categories.map((category) => <option key={category} value={category}>{category}</option>)}
            </select>
          </label>
          <div className="dataset-control-actions">
            <button
              className="primary-button dataset-run-button"
              disabled={isRunning}
              onClick={runEvaluation}
              type="button"
            >
              {isRunning ? `Running ${completed}/${evaluationExamples.length}…` : `Run ${evaluationExamples.length} tests`}
              {!isRunning && <span aria-hidden="true">↗</span>}
            </button>
            {results.length > 0 && (
              <button className="dataset-reset-button" disabled={isRunning} onClick={resetResults} type="button">
                Clear results
              </button>
            )}
          </div>
        </div>
      </section>

      <p className="dataset-api-note">
        Uses the model and generation settings configured by your FastAPI endpoint. Requests run one
        at a time; few-shot mode includes the six training examples in each prompt.
      </p>

      {error && <div className="dataset-error" role="alert">{error}</div>}

      {results.length > 0 && (
        <section className="dataset-metrics" aria-label="Evaluation summary">
          <article className="dataset-metric-card">
            <span>Accuracy</span>
            <strong>{Math.round((correctCount / results.length) * 100)}%</strong>
            <small>{correctCount} correct of {results.length} completed</small>
          </article>
          <article className="dataset-metric-card">
            <span>Valid category responses</span>
            <strong>{Math.round((validCount / results.length) * 100)}%</strong>
            <small>{validCount} outputs matched an allowed label</small>
          </article>
          {categoryMetrics
            .filter((metric) => metric.total > 0)
            .map((metric) => (
              <article className="dataset-metric-card category-metric" key={metric.category}>
                <span>{metric.category}</span>
                <strong>{metric.correct}/{metric.total}</strong>
                <small>correct</small>
              </article>
            ))}
        </section>
      )}

      <section className="dataset-results-panel">
        <div className="dataset-results-heading">
          <div>
            <span className="section-kicker">EVALUATION EXAMPLES</span>
            <h2>Model results</h2>
          </div>
          {results.length > 0 && (
            <span className="dataset-progress">
              {results.length} of {evaluationExamples.length} completed
            </span>
          )}
        </div>

        {results.length === 0 ? (
          <div className="dataset-empty-state">
            <span aria-hidden="true">✦</span>
            <strong>Ready to evaluate</strong>
            <p>Choose zero-shot or few-shot, then run the evaluation to see predictions beside the expected labels.</p>
          </div>
        ) : (
          <div className="dataset-table-wrap">
            <table className="dataset-results-table">
              <thead>
                <tr>
                  <th scope="col">Ticket</th>
                  <th scope="col">Expected</th>
                  <th scope="col">Model label</th>
                  <th scope="col">Result</th>
                  <th scope="col">Raw response</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.id}>
                    <td>
                      <span className="dataset-ticket-id">{result.id}</span>
                      <span className="dataset-ticket-input">{result.input}</span>
                    </td>
                    <td>{result.expected}</td>
                    <td>{result.predicted ?? 'Invalid label'}</td>
                    <td>
                      <span className={`dataset-result-badge ${result.correct ? 'correct' : 'incorrect'}`}>
                        {result.correct ? 'Correct' : 'Incorrect'}
                      </span>
                    </td>
                    <td className="dataset-raw-response">{result.rawResponse}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

export default PromptLabPage
