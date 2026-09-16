import { useState, useEffect } from 'react';
import { Modal, Form, Input, Button, Alert, Radio, Typography, Select } from 'antd'; 
import { useAppContext } from '../context/AppContext';

const { Text } = Typography;

export const LLMConfigModal = ({ visible, onClose, onConfigured }) => {
  const [form] = Form.useForm();
  const { llmConfig, updateLlmConfig } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [provider, setProvider] = useState(llmConfig.provider);
  const [availableModels, setAvailableModels] = useState([]); 
  const [testStatus, setTestStatus] = useState(null);

const fetchModels = async (currentProvider, currentBaseUrl) => {
  if (!currentProvider) return;

  try {
    const params = new URLSearchParams({ 
      provider: currentProvider,
      base_url: currentBaseUrl || '' 
    });
    
    const response = await fetch(`/config/llm/models?${params.toString()}`);
    
    const contentType = response.headers.get("content-type");
    if (response.ok && contentType && contentType.includes("application/json")) {
      const data = await response.json();
      setAvailableModels(data.models || []);
    } else {
      setAvailableModels([]); 
    }
  } catch (error) {
    console.error("Connection failed:", error);
    setAvailableModels([]);
  }
};

  useEffect(() => {
    if (visible) {
      const initialValues = {
        provider: llmConfig.provider,
        // Wrap the initial model string in an array:
        model: llmConfig.model ? [llmConfig.model] : [], 
        baseUrl: llmConfig.baseUrl || (llmConfig.provider === 'openai' 
          ? 'https://api.openai.com/v1' 
          : 'http://localhost:11434'),
        apiKey: llmConfig.apiKey
      };
      form.setFieldsValue(initialValues);
      fetchModels(initialValues.provider, initialValues.baseUrl);
    }
  }, [visible, llmConfig, form]);

  const handleProviderChange = (value) => {
    setProvider(value);
    let newUrl = '';
    if (value === 'openai') newUrl = 'https://api.openai.com/v1';
    else if (value === 'ollama') newUrl = 'http://localhost:11434';
    else if (value === 'lm_studio') newUrl = 'http://localhost:1234/v1';
    
    form.setFieldValue('baseUrl', newUrl);
    fetchModels(value, newUrl);
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setTestStatus(null);

      const response = await fetch('/config/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: values.provider,
          model: values.model,
          api_key: values.apiKey,
          base_url: values.baseUrl
        })
      });

      if (response.ok) {
        updateLlmConfig({
          provider: values.provider,
          model: values.model,
          baseUrl: values.baseUrl,
          apiKey: values.apiKey,
          configured: true
        });
        setTestStatus({ type: 'success', message: 'LLM configured successfully!' });
        setTimeout(() => {
          onConfigured?.();
          onClose();
        }, 1000);
      } else {
        const error = await response.json();
        setTestStatus({ type: 'error', message: error.detail || 'Configuration failed' });
      }
    } catch (error) {
      setTestStatus({ type: 'error', message: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title="LLM Configuration"
      open={visible}
      onOk={handleSave}
      onCancel={onClose}
      confirmLoading={loading}
      width={600}
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
            onChange={(e) => fetchModels(provider, e.target.value)}
          />
        </Form.Item>


        <Form.Item
          name="apiKey"
          label="API Key"
          rules={[{ required: provider === 'openai' }]}
        >
          <Input.Password placeholder={provider === 'openai' ? "sk-..." : "Optional for local providers"} />
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

        <Form.Item 
          name="model" 
          label="Model" 
          rules={[{ required: true }]}
          getValueProps={(value) => ({

            value: value ? (Array.isArray(value) ? value : [value]) : [],
          })}
          normalize={(value) => {

            return Array.isArray(value) ? value[0] : value;
          }}
        >
          <Select
            mode="tags"
            placeholder="Select or type model name"
            options={availableModels.map(m => ({ label: m, value: m }))}
            maxCount={1} 
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};