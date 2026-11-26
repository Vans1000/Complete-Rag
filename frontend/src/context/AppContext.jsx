import { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();

const COLLECTION_STORAGE_KEY = 'current_collection';

export const AppProvider = ({ children }) => {
  const [mode, setMode] = useState('local');
  const [webIngest, setWebIngest] = useState(false);

  const [currentCollection, setCurrentCollection] = useState(() => {
    try {
      return localStorage.getItem(COLLECTION_STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });

  const [treeRagEnabled, setTreeRagEnabled] = useState(false);

  const [llmConfig, setLlmConfig] = useState(() => {
    const saved = localStorage.getItem('llm_config');
    return saved ? JSON.parse(saved) : {
      provider: 'openai',
      model: 'gpt-4o-mini',
      baseUrl: '',
      apiKey: '',
      configured: false
    };
  });

  const [tokenizerConfig, setTokenizerConfig] = useState(() => {
    const saved = localStorage.getItem('tokenizer_config');
    return saved ? JSON.parse(saved) : {
      denseModel: 'clip-ViT-B-32',
      textDenseModel: 'BAAI/bge-small-en-v1.5',
      sparseModel: 'naver/splade-v3',
      crossEncoderModel: 'Alibaba-NLP/gte-reranker-modernbert-base',
      visionRerankModel: 'nvidia/llama-nemotron-rerank-vl-1b-v2',
      captionModel: 'vikhyatk/moondream2'
    };
  });

  useEffect(() => {
    localStorage.setItem('llm_config', JSON.stringify(llmConfig));
  }, [llmConfig]);

  useEffect(() => {
    localStorage.setItem('tokenizer_config', JSON.stringify(tokenizerConfig));
  }, [tokenizerConfig]);

  useEffect(() => {
    try {
      if (currentCollection) {
        localStorage.setItem(COLLECTION_STORAGE_KEY, currentCollection);
      } else {
        localStorage.removeItem(COLLECTION_STORAGE_KEY);
      }
    } catch (e) {
      console.warn('[AppContext] Failed to persist collection:', e);
    }
  }, [currentCollection]);

  const updateLlmConfig = (newConfig) => {
    setLlmConfig(prev => ({ ...prev, ...newConfig }));
  };

  const updateTokenizerConfig = (newConfig) => {
    setTokenizerConfig(prev => ({ ...prev, ...newConfig }));
  };

  return (
    <AppContext.Provider value={{
      mode, setMode,
      webIngest, setWebIngest,
      currentCollection, setCurrentCollection,
      llmConfig, updateLlmConfig,
      tokenizerConfig, updateTokenizerConfig,
      treeRagEnabled,
      setTreeRagEnabled,
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);