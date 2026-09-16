import { useState, useEffect } from 'react';
import { Modal, Form, Input, Button, Alert, Select, Typography } from 'antd';
import { useAppContext } from '../context/AppContext';

const { Text } = Typography;

const CAPTION_OPTIONS = [
  { label: 'None (Fastest)', value: 'None' },
  { label: 'Moondream2', value: 'vikhyatk/moondream2' },
  { label: 'BLIP Salesforce', value: 'Salesforce/blip-image-captioning-base' },
  { label: 'Florence-2', value: 'microsoft/Florence-2-large' },
  { label: 'LLaVA', value: 'llava-hf/llava-1.5-7b-hf' }
];

export const TokenizerConfigModal = ({ visible, onClose }) => {
  const [form] = Form.useForm();
  const { tokenizerConfig, updateTokenizerConfig } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    if (visible) {
      form.setFieldsValue({
        denseModel: tokenizerConfig.denseModel,
        sparseModel: tokenizerConfig.sparseModel,
        crossEncoderModel: tokenizerConfig.crossEncoderModel,
        visionRerankModel: tokenizerConfig.visionRerankModel,
        captionModel: tokenizerConfig.captionModel
      });
      // Also fetch current server-side config
      fetch('/config/tokenizer')
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data) {
            form.setFieldsValue({
              denseModel: data.dense_model,
              sparseModel: data.sparse_model,
              crossEncoderModel: data.cross_encoder_model,
              visionRerankModel: data.vision_rerank_model,
              captionModel: data.caption_model
            });
          }
        })
        .catch(() => {});
    }
  }, [visible, tokenizerConfig, form]);

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setStatus(null);

      const response = await fetch('/config/tokenizer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dense_model: values.denseModel,
          sparse_model: values.sparseModel,
          cross_encoder_model: values.crossEncoderModel,
          vision_rerank_model: values.visionRerankModel,
          caption_model: values.captionModel
        })
      });

      if (response.ok) {
        updateTokenizerConfig({
          denseModel: values.denseModel,
          sparseModel: values.sparseModel,
          crossEncoderModel: values.crossEncoderModel,
          visionRerankModel: values.visionRerankModel,
          captionModel: values.captionModel
        });
        setStatus({ type: 'success', message: 'Tokenizer models updated! Restart ingestion for changes to take full effect.' });
      } else {
        const error = await response.json();
        setStatus({ type: 'error', message: error.detail || 'Update failed' });
      }
    } catch (error) {
      setStatus({ type: 'error', message: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title="Embedding & Caption Models"
      open={visible}
      onOk={handleSave}
      onCancel={onClose}
      confirmLoading={loading}
      width={600}
      okText="Save Configuration"
    >
      <Form form={form} layout="vertical">
        <Form.Item name="denseModel" label="Dense Embedding Model" rules={[{ required: true }]}>
          <Input placeholder="e.g. clip-ViT-B-32" />
        </Form.Item>

        <Form.Item name="sparseModel" label="Sparse Model (SPLADE)" rules={[{ required: true }]}>
          <Input placeholder="e.g. naver/splade-v3" />
        </Form.Item>

        <Form.Item name="crossEncoderModel" label="Cross-Encoder Reranker" rules={[{ required: true }]}>
          <Input placeholder="e.g. cross-encoder/ms-marco-MiniLM-L-12-v2" />
        </Form.Item>

        <Form.Item name="visionRerankModel" label="Vision Reranker" rules={[{ required: true }]}>
          <Input placeholder="e.g. nvidia/llama-nemotron-rerank-vl-1b-v2" />
        </Form.Item>

        <Form.Item name="captionModel" label="Image Caption Model" rules={[{ required: true }]}>
          <Select
            showSearch
            allowClear
            placeholder="Select caption model or None"
            options={CAPTION_OPTIONS}
          />
        </Form.Item>

        {status && (
          <Alert
            message={status.type === 'success' ? 'Success' : 'Error'}
            description={status.message}
            type={status.type}
            showIcon
            className="test-status-alert"
          />
        )}

        <Text type="secondary" style={{ display: 'block', marginTop: 16, fontSize: 12 }}>
          Changing these models reloads them into memory. Existing vectors remain in Qdrant.
        </Text>
      </Form>
    </Modal>
  );
};