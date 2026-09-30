import { useState, useEffect, useCallback } from 'react';
import { Modal, Form, Input, Button, Alert, Radio, Typography, Select, Space, Tag, Tooltip } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { useAppContext } from '../context/AppContext';

const { Text } = Typography;

export const LLMConfigModal = ({ visible, onClose, onConfigured }) => {
  const [form] = Form.useForm();
  const { llmConfig, updateLlmConfig } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [provider, setProvider] = useState(llmConfig.provider);
  const [availableModels, setAvailableModels] = useState([]);
  const [modelSource, setModelSource] = useState(null); // 'live' | 'fallback'
  const [testStatus, setTestStatus] = useState(null);

  const fetchModels = useCallback(async (currentProvider, currentBaseUrl, currentApiKey) => {
    if (!currentProvider) return;
    setLoadingModels(true);
    try {
      const params = new URLSearchParams({
        provider: currentProvider,
        base_url: currentBaseUrl || '',
      });
      if (currentApiKey) params.append('api_key', currentApiKey);

      const response = await fetch(`/config/llm/models?${params.toString()}`);
      const ct = response.headers.get('content-type') || '';
      if (response.ok && ct.includes('application/json')) {
        const data = await response.json();
        setAvailableModels(data.models || []);
        setModelSource(data.source || null);
      } else {
        setAvailableModels([]);
        setModelSource(null);
      }
    } catch (err) {
      console.error('Model fetch failed:', err);
      setAvailableModels([]);
      setModelSource(null);
    } finally {
      setLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    if (!visible) return;
    const initial = {
      provider: llmConfig.provider || 'openai',
      model: llmConfig.model || undefined,
      baseUrl: llmConfig.baseUrl || (llmConfig.provider === 'ollama'
        ? 'http://localhost:11434'
        : 'https://api.openai.com/v1'),
      apiKey: llmConfig.apiKey || '',
    };
    form.setFieldsValue(initial);
    fetchModels(initial.provider, initial.baseUrl, initial.apiKey);
  }, [visible, llmConfig, form, fetchModels]);

  const handleProviderChange = (value) => {
    setProvider(value);
    const newUrl =
      value === 'openai' ? 'https://api.openai.com/v1' :
      value === 'ollama' ? 'http://localhost:11434' :
      value === 'lm_studio' ? 'http://localhost:1234/v1' : '';

    form.setFieldsValue({ baseUrl: newUrl, model: undefined, apiKey: '' });
    setAvailableModels([]);
    setModelSource(null);
    fetchModels(value, newUrl, '');
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setTestStatus(null);

      const selectedModel = Array.isArray(values.model) ? values.model[0] : values.model;

      const response = await fetch('/config/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: values.provider,
          model: selectedModel,
          api_key: values.apiKey || null,
          base_url: values.baseUrl || null,
        }),
      });

      if (response.ok) {
        updateLlmConfig({
          provider: values.provider,
          model: selectedModel,       
          baseUrl: values.baseUrl,
          apiKey: values.apiKey,
          configured: true,
        });
        setTestStatus({ type: 'success', message: 'LLM configured successfully!' });
        setTimeout(() => { onConfigured?.(); onClose(); }, 800);
      } else {
        const error = await response.json().catch(() => ({}));
        const msg = typeof error.detail === 'object'
          ? JSON.stringify(error.detail)
          : (error.detail || 'Configuration failed');
        setTestStatus({ type: 'error', message: msg });
      }
    } catch (err) {
      setTestStatus({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  const isLocalProvider = provider === 'ollama' || provider === 'lm_studio' || provider === 'custom';

  return (
    <Modal
      title="LLM Configuration"
      open={visible}
      onOk={handleSave}
      onCancel={onClose}
      confirmLoading={loading}
      width={620}
      okText="Save & Test Configuration"
    >
      <Form form={form} layout="vertical">
        <Form.Item name="provider" label="Provider" rules={[{ required: true }]}>
          <Radio.Group onChange={(e) => handleProviderChange(e.target.value)}>
            <Radio.Button value="openai">OpenAI</Radio.Button>
            <Radio.Button value="ollama">Ollama</Radio.Button>
            <Radio.Button value="lm_studio">LM Studio</Radio.Button>
            <Radio.Button value="custom">Custom (OpenAI-Compatible)</Radio.Button>
          </Radio.Group>
        </Form.Item>

        <Form.Item name="baseUrl" label="Base URL" rules={[{ required: true }]}>
          <Input
            placeholder="e.g. http://localhost:1234/v1"
            onBlur={(e) => fetchModels(provider, e.target.value, form.getFieldValue('apiKey'))}
          />
        </Form.Item>

        <Form.Item
          name="apiKey"
          label="API Key"
          rules={[{ required: provider === 'openai' }]}
          extra={isLocalProvider
            ? 'Leave blank for local servers — a placeholder will be sent automatically.'
            : undefined}
        >
          <Input.Password
            placeholder={provider === 'openai' ? 'sk-...' : 'Optional for local providers'}
            onBlur={(e) => fetchModels(provider, form.getFieldValue('baseUrl'), e.target.value)}
          />
        </Form.Item>

        <Form.Item
          name="model"
          label={
            <Space>
              <span>Model</span>
              {modelSource === 'live' && <Tag color="green">detected</Tag>}
              {modelSource === 'fallback' && <Tag color="orange">fallback list</Tag>}
              <Tooltip title="Re-query the provider for its model list">
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined spin={loadingModels} />}
                  onClick={() => fetchModels(
                    provider,
                    form.getFieldValue('baseUrl'),
                    form.getFieldValue('apiKey'),
                  )}
                />
              </Tooltip>
            </Space>
          }
          rules={[{ required: true }]}
          getValueProps={(value) => ({
            value: value ? (Array.isArray(value) ? value : [value]) : [],
          })}
          normalize={(value) => (Array.isArray(value) ? value[0] : value)}
        >
          <Select
            mode="tags"
            showSearch
            loading={loadingModels}
            placeholder={loadingModels ? 'Detecting models…' : 'Select or type a model name'}
            options={availableModels.map(m => ({ label: m, value: m }))}
            maxCount={1}
            notFoundContent={loadingModels ? 'Loading…' : 'No models detected — type one manually'}
          />
        </Form.Item>

        {testStatus && (
          <Alert
            title={testStatus.type === 'success' ? 'Success' : 'Error'}
            description={testStatus.message}
            type={testStatus.type}
            showIcon
            className="test-status-alert"
          />
        )}
      </Form>
    </Modal>
  );
};