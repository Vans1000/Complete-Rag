import { useState, useRef, useEffect, useCallback } from 'react';
import { Layout, Input, Button, message, Upload, Card, List, Typography, Spin, Badge, Progress, Tooltip } from 'antd';
import { InboxOutlined, SendOutlined, LoadingOutlined, FileOutlined, CloseCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { CollectionSelector } from '../components/CollectionSelector';
import { ModeToggle } from '../components/ModeToggle';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { useAppContext } from '../context/AppContext';
import { Switch } from 'antd';

const { Sider, Content } = Layout;
const { Text } = Typography;
const { Dragger } = Upload;

const HISTORY_PREFIX = 'rag_chat_history_';
const historyKey = (collection) => `${HISTORY_PREFIX}${collection || 'default'}`;

const loadHistory = (collection) => {
  try {
    const raw = localStorage.getItem(historyKey(collection));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn('[Chat] Failed to load history:', e);
    return [];
  }
};

const saveHistory = (collection, messages) => {
  try {
    localStorage.setItem(historyKey(collection), JSON.stringify(messages));
  } catch (e) {
    console.warn('[Chat] Failed to save history:', e);
  }
};

export const ChatPage = () => {
  const [messages, setMessages] = useState(() => loadHistory(null));
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);

  const {
    mode, webIngest, llmConfig,
    currentCollection, setCurrentCollection,
    treeRagEnabled, setTreeRagEnabled,
  } = useAppContext();

  const [uploadingFiles, setUploadingFiles] = useState([]);
  const messagesEndRef = useRef(null);

  const skipNextSaveRef = useRef(false);
  const collectionRef = useRef(currentCollection);

  useEffect(() => {
    if (collectionRef.current === currentCollection) return;
    collectionRef.current = currentCollection;
    skipNextSaveRef.current = true;
    setMessages(loadHistory(currentCollection));
  }, [currentCollection]);

  useEffect(() => {
    if (skipNextSaveRef.current) {
      skipNextSaveRef.current = false;
      return;
    }
    if (messages.length === 0 && !localStorage.getItem(historyKey(collectionRef.current))) {
      return; 
    }
    saveHistory(collectionRef.current, messages);
  }, [messages]);

  useEffect(() => {
    if (collectionRef.current !== currentCollection) return;
    saveHistory(currentCollection, messages);
  }, [currentCollection]);

  const skipFirstSyncRef = useRef(true);

  useEffect(() => {
    if (skipFirstSyncRef.current) {
      skipFirstSyncRef.current = false;
      fetch('/config/tree_rag')
        .then(r => r.json())
        .then(data => {
          if (typeof data.enabled === 'boolean' && data.enabled !== treeRagEnabled) {
            setTreeRagEnabled(data.enabled);
          }
        })
        .catch(() => { });
      return;
    }

    fetch('/config/tree_rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !!treeRagEnabled }),
    })
      .then(r => r.ok ? null : r.json().then(e =>
        message.warning(e.detail || 'Failed to toggle TreeRAG on the server')
      ))
      .catch(e => console.warn('[TreeRAG] Sync failed:', e));
  }, [treeRagEnabled]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleClearChat = () => {
    setMessages([]);
    try {
      localStorage.removeItem(historyKey(collectionRef.current));
    } catch (e) {
      /* ignore */
    }
    message.success('Chat cleared');
  };

  const handleSend = async () => {
    if (!input.trim() || streaming) return;

    const userMsg = { role: 'user', content: input, id: Date.now() };
    const assistantMsgId = Date.now() + 1;

    setMessages(prev => [
      ...prev,
      userMsg,
      { role: 'assistant', content: '', id: assistantMsgId, sources: [] },
    ]);
    setInput('');
    setStreaming(true);

    try {
      const useWeb = mode === 'web' || mode === 'force_web';
      const forceWeb = mode === 'force_web';

      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          use_web_search: useWeb,
          force_web: forceWeb,
          ingest_web: webIngest,
        }),
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let resultText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6).trim();
          if (dataStr === '[DONE]') continue;

          try {
            const data = JSON.parse(dataStr);
            if (data.chunk) {
              resultText += data.chunk;
              setMessages(prev => prev.map(msg =>
                msg.id === assistantMsgId ? { ...msg, content: resultText } : msg
              ));
            }
          } catch (e) {
          }
        }
      }
    } catch (error) {
      message.error('Chat error: ' + error.message);
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId ? { ...msg, content: 'Error: ' + error.message } : msg
      ));
    } finally {
      setStreaming(false);
    }
  };

   const pollUploadProgress = useCallback((uploadId, fileId) => {
    console.log('[Upload] Starting progress poll for', uploadId);
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/ingest/progress/${uploadId}`);
        if (!res.ok) {
          clearInterval(interval);
          setUploadingFiles(prev => prev.map(f => f.id === fileId ? { ...f, status: 'error' } : f));
          return;
        }
        const data = await res.json();
        console.log('[Upload] Poll response:', data);

        setUploadingFiles(prev => prev.map(f =>
          f.id === fileId ? { ...f, status: data.status, progress: data.progress } : f
        ));

        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(interval);
          if (data.status === 'completed') {
            message.success(`Processed ${data.filename || 'file'}`);
            setTimeout(() => {
              setUploadingFiles(prev => prev.filter(f => f.id !== fileId));
            }, 3000);
          } else {
            message.error(`Ingestion failed: ${data.message || 'unknown error'}`);
            setUploadingFiles(prev =>
              prev.map(f => f.id === fileId ? { ...f, status: 'error' } : f)
            );
          }
        }
      } catch (e) {
        console.error('[Upload] Poll failed:', e);
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (file) => {
    console.log('[Upload] beforeUpload called with', file.name, file.type, file.size);
    const fileId = Date.now() + Math.random();
    const newFile = { id: fileId, name: file.name, status: 'uploading', progress: 0 };

    setUploadingFiles(prev => [...prev, newFile]);

    const formData = new FormData();
    formData.append('file', file);
    if (currentCollection) {
      formData.append('collection', currentCollection);
    }
    formData.append('tree_rag_enabled', treeRagEnabled ? 'true' : 'false');
    try {
      const response = await fetch('/ingest/file', {
        method: 'POST',
        body: formData,
      });

      console.log('[Upload] Server responded', response.status, response.statusText);

      if (response.ok) {
        const data = await response.json();
        console.log('[Upload] Server JSON:', data);
        setUploadingFiles(prev =>
          prev.map(f => f.id === fileId ? { ...f, status: 'processing', progress: 50, uploadId: data.upload_id } : f)
        );
        message.success(`Uploaded ${file.name} for processing`);

        if (data.upload_id) {
          pollUploadProgress(data.upload_id, fileId);
        }
      } else {
        const errText = await response.text();
        console.error('[Upload] Server error body:', errText);
        throw new Error(`Server ${response.status}: ${errText}`);
      }
    } catch (error) {
      console.error('[Upload] Fetch failed:', error);
      setUploadingFiles(prev =>
        prev.map(f => f.id === fileId ? { ...f, status: 'error', progress: 0 } : f)
      );
      message.error('Upload failed: ' + error.message);
    }
    return false;
  };

  const removeUploadingFile = (fileId) => {
    setUploadingFiles(prev => prev.filter(f => f.id !== fileId));
  };

  return (
    <Layout className="chat-layout">
      <Sider width={320} className="chat-sider">
        <Card size="small" title="Collection" className="sider-card">
          <CollectionSelector
            value={currentCollection}
            onChange={setCurrentCollection}
          />
        </Card>

        <Card size="small" title="Search Mode" className="sider-card">
          <ModeToggle />
        </Card>

        <Card size="small" title="Active LLM" className="sider-card">
          <div className="llm-info">
            <div><Text strong>Provider:</Text> <span className="llm-value">{llmConfig.provider}</span></div>
            <div><Text strong>Model:</Text> <span className="llm-value">{llmConfig.model}</span></div>
            <div className="llm-url">{llmConfig.baseUrl}</div>
          </div>
        </Card>

        <Card title="TreeRag" size="small" className="sider-card">
          <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Switch checked={treeRagEnabled} onChange={setTreeRagEnabled} />
            <Text>Enable TreeRAG (hierarchical summarization)</Text>
          </div>
        </Card>

        <Card size="small" title="Upload Documents" className="sider-card">
          <Dragger
            beforeUpload={handleFileUpload}
            showUploadList={false}
            multiple={true}
            className="upload-dragger"
          >
            <p className="upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="upload-text">Drop docs here or click to upload</p>
          </Dragger>

          {uploadingFiles.length > 0 && (
            <div className="uploading-files">
              <Text type="secondary" className="uploading-title">Uploading Files:</Text>
              {uploadingFiles.map(file => (
                <div key={file.id} className="uploading-file-item" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FileOutlined className="file-icon" />
                    <span className="file-name" title={file.name}>{file.name}</span>
                    {file.status === 'uploading' && (
                      <Spin indicator={<LoadingOutlined spin className="loading-icon" />} size="small" />
                    )}
                    {file.status === 'completed' && <Badge status="success" text="Done" />}
                    {file.status === 'error' && <Badge status="error" text="Failed" />}
                    <Button
                      type="text"
                      size="small"
                      icon={<CloseCircleOutlined />}
                      onClick={() => removeUploadingFile(file.id)}
                      className="remove-btn"
                    />
                  </div>
                  {(file.status === 'uploading' || file.status === 'processing') && (
                    <Progress
                      percent={file.progress}
                      size="small"
                      status={file.status === 'processing' ? 'active' : 'normal'}
                      style={{ marginTop: 4 }}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </Sider>

      <Layout>
        <Content className="chat-content">
          <div className="chat-toolbar">
            <Text type="secondary" className="chat-toolbar-title">
              {currentCollection ? `Collection: ${currentCollection}` : 'No collection selected'}
              {messages.length > 0 && ` · ${messages.filter(m => m.role === 'user').length} message(s)`}
            </Text>
            <Tooltip title="Delete the saved conversation for this collection">
              <Button
                size="small"
                icon={<DeleteOutlined />}
                onClick={handleClearChat}
                disabled={messages.length === 0 || streaming}
              >
                Clear chat
              </Button>
            </Tooltip>
          </div>

          <div className="messages-container">
            {messages.length === 0 && (
              <div className="empty-chat">
                <div className="empty-chat-icon">💬</div>
                <Text type="secondary" className="empty-chat-text">
                  Start a conversation by typing a message below
                </Text>
              </div>
            )}

            <List
              dataSource={messages}
              renderItem={(msg) => {
                let displayContent = msg.content;
                let sources = msg.sources || [];

                if (typeof msg.content === 'string' && msg.content.includes('"answer":')) {
                  try {
                    const parsed = JSON.parse(msg.content);
                    displayContent = parsed.answer;
                    sources = parsed.sources || [];
                  } catch (e) { /* not JSON, ignore */ }
                }

                return (
                  <div className={`message-wrapper ${msg.role}`}>
                    <div className={`message-bubble ${msg.role}`}>
                      {msg.role === 'assistant' ? (
                        displayContent
                          ? <MarkdownRenderer content={displayContent} />
                          : <span className="typing-cursor">▍</span>
                      ) : (
                        <div className="text-content" style={{ whiteSpace: 'pre-wrap' }}>
                          {displayContent}
                        </div>
                      )}
                      {sources.length > 0 && (
                        <div className="sources-container">
                          <div className="sources-title">Sources:</div>
                          {sources.map((s, idx) => {
                            let rawSource = typeof s === 'string' ? s : (s.source || '');

                      
                            let sanitizedContent = rawSource;

                            // 1. Remove bullet points like "o " or "◦ " at the start of lines
                            sanitizedContent = sanitizedContent.replace(/^\s*(?:o|◦|[-•])\s+/gm, '');

                            // 2. Join lines that end with '=' with the next line (handles "y = \n (v - o_y)s_y")
                            sanitizedContent = sanitizedContent.replace(/([a-zA-Z]\s*=)\s*\n\s*(.*?)\s*(?=\n|$)/g, '$1 $2');

                            // 3. Remove trailing PDF junk like '").' or '")' or '.' at the end of lines
                            sanitizedContent = sanitizedContent.replace(/["')\s]*\.?\s*$/gm, '');

                            // 4. Wrap standalone equations in $$...$$ if they contain '='
                            sanitizedContent = sanitizedContent.split('\n').map(line => {
                              const trimmed = line.trim();
                              // If line contains '=' and doesn't already start with '$', wrap it in block math
                              if (trimmed.includes('=') && !trimmed.startsWith('$') && trimmed.length > 3) {
                                return `$$${trimmed}$$`;
                              }
                              return line;
                            }).join('\n');
                            printf('Sanitized source content:', sanitizedContent);
                            return (
                              <MarkdownRenderer 
                                key={idx} 
                                className="source-item" 
                                content={sanitizedContent} 
                              />
                            );
                          })}
                        </div>
                      )}

                    </div>
                  </div>
                );
              }}
            />
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <Input.TextArea
              value={input}
              onChange={e => setInput(e.target.value)}
              onPressEnter={e => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask a question...  (LaTeX: $x^2$ or $$\int_0^1 x\,dx$$)"
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={streaming}
              className="chat-input"
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              loading={streaming}
              className="send-btn"
            >
              Send
            </Button>
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};