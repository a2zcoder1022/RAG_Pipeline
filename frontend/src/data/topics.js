export const topics = [
  {
    id: 'tokens',
    title: 'Tokens & tokenization',
    category: 'Foundations',
    level: 'Beginner',
    minutes: 6,
    icon: 'Aa',
    color: 'violet',
    summary: 'See how text is split into the pieces a language model can actually process.',
    lesson: {
      intro:
        'A language model does not read words the way we do. A tokenizer turns text into a sequence of tokens—often whole words, word pieces, punctuation, or whitespace.',
      points: [
        'The model receives token IDs, not raw text.',
        'Common word pieces often become a single token; rare words may split into several.',
        'The context window and many API limits are measured in tokens.',
      ],
      example: '“unbelievable!”  →  [“un”, “believ”, “able”, “!”]',
      note: 'This is an illustrative split. Actual token boundaries depend on the tokenizer.',
    },
  },
  {
    id: 'embeddings',
    title: 'Embeddings',
    category: 'Foundations',
    level: 'Beginner',
    minutes: 7,
    icon: '✳',
    color: 'blue',
    summary: 'Learn how meaning can be represented as numbers and compared by similarity.',
    lesson: {
      intro:
        'An embedding maps a piece of text to a list of numbers. Texts used in similar contexts often have vectors that are close together.',
      points: [
        'Embeddings make semantic search possible.',
        'A vector database can retrieve passages related to a question.',
        'Similarity is a useful signal, not proof that two statements mean exactly the same thing.',
      ],
      example: '“How do I change my password?”  →  [0.12, -0.08, 0.41, …]',
      note: 'Real embeddings are much longer; the values here are simplified.',
    },
  },
  {
    id: 'attention',
    title: 'Attention',
    category: 'Foundations',
    level: 'Intermediate',
    minutes: 9,
    icon: '◉',
    color: 'orange',
    summary: 'Explore how a model weighs relationships between tokens in context.',
    lesson: {
      intro:
        'Attention lets each token use information from other tokens. It helps a model represent which parts of the surrounding context matter for the next prediction.',
      points: [
        'Self-attention relates tokens within the same sequence.',
        'Attention is computed from query, key, and value representations.',
        'It is one important mechanism in Transformers, not a complete explanation of understanding.',
      ],
      example: '“The dog chased the ball because it was moving.”',
      note: 'Attention can help connect “it” to context, but interpreting attention weights takes care.',
    },
  },
  {
    id: 'transformers',
    title: 'The Transformer',
    category: 'Models',
    level: 'Intermediate',
    minutes: 10,
    icon: '▦',
    color: 'green',
    summary: 'Understand the architecture behind most modern language models.',
    lesson: {
      intro:
        'The Transformer processes token representations through layers that combine attention and feed-forward computations.',
      points: [
        'Text is converted into token IDs and then vectors.',
        'Transformer layers update those vectors using context.',
        'A language model uses its final representation to predict likely next tokens.',
      ],
      example: 'Text → tokens → embeddings → Transformer layers → next-token scores',
      note: 'Different model families vary in architecture, training, and how they generate output.',
    },
  },
  {
    id: 'prompting',
    title: 'Prompt engineering',
    category: 'Building',
    level: 'Beginner',
    minutes: 8,
    icon: '⌘',
    color: 'pink',
    summary: 'Write clearer instructions, provide useful context, and specify the output you need.',
    lesson: {
      intro:
        'A prompt is the input and instructions given to a model. Good prompts make the task, context, constraints, and desired output clear.',
      points: [
        'State the task and audience directly.',
        'Include relevant context and examples.',
        'Ask for a useful format, then test prompts against real cases.',
      ],
      example: '“Summarize this report in 3 bullets for a new analyst. Use only the text provided.”',
      note: 'Prompting can improve results, but it does not guarantee correctness.',
    },
  },
  {
    id: 'rag',
    title: 'Retrieval-augmented generation',
    category: 'Building',
    level: 'Intermediate',
    minutes: 11,
    icon: '⌕',
    color: 'teal',
    summary: 'Connect a model to relevant documents so answers can use external information.',
    lesson: {
      intro:
        'RAG retrieves relevant passages from a knowledge source and places them in the model’s prompt before it generates a response.',
      points: [
        'Split documents into useful chunks and index their embeddings.',
        'Embed a user question and retrieve similar chunks.',
        'Give retrieved context to the model and make sources visible to the user.',
      ],
      example: 'Question → retrieve passages → answer from those passages → show sources',
      note: 'RAG can provide evidence, but retrieval quality and answer grounding still need evaluation.',
    },
  },
  {
    id: 'fine-tuning',
    title: 'Fine-tuning',
    category: 'Models',
    level: 'Intermediate',
    minutes: 8,
    icon: '✎',
    color: 'yellow',
    summary: 'Find out when examples can adapt a model—and when retrieval is a better fit.',
    lesson: {
      intro:
        'Fine-tuning continues training a pretrained model on a targeted dataset to influence its behavior or task performance.',
      points: [
        'Use representative, carefully prepared examples.',
        'Fine-tuning changes model parameters; RAG supplies external context at run time.',
        'Measure performance on held-out examples before relying on a tuned model.',
      ],
      example: 'Base model + curated task examples → adapted model behavior',
      note: 'Fine-tuning is not automatically the best way to give a model new facts.',
    },
  },
  {
    id: 'inference',
    title: 'Inference & sampling',
    category: 'Models',
    level: 'Beginner',
    minutes: 6,
    icon: '↗',
    color: 'purple',
    summary: 'See how temperature, top-p, and token limits affect generated responses.',
    lesson: {
      intro:
        'Inference is the process of using a trained model to produce output. Generation settings influence which next token is selected and how long the response can be.',
      points: [
        'Temperature adjusts how concentrated or varied token selection is.',
        'Top-p limits sampling to a nucleus of likely token options.',
        'A maximum output-token setting caps response length.',
      ],
      example: 'Prompt + model + generation settings → generated tokens',
      note: 'Parameter behavior can differ between model providers and implementations.',
    },
  },
  {
    id: 'evaluation',
    title: 'Evaluating LLMs',
    category: 'Building',
    level: 'Intermediate',
    minutes: 9,
    icon: '✓',
    color: 'red',
    summary: 'Build a repeatable way to check quality, reliability, and regressions.',
    lesson: {
      intro:
        'Evaluation tests how well a model or LLM-powered application performs on tasks that matter to its users.',
      points: [
        'Create a representative set of questions and expected behavior.',
        'Measure factuality, relevance, safety, and latency as appropriate.',
        'Re-run the same evaluations after changing prompts, models, or retrieval.',
      ],
      example: 'Test set → run system → inspect scores and examples → improve → repeat',
      note: 'Automated scores are useful signals; human review is important for nuanced tasks.',
    },
  },
]

export const categories = ['All topics', 'Foundations', 'Models', 'Building']
