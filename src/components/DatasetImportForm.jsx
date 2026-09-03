import { useState } from 'react';
import { Form, Input, Button, InputNumber, message, Card, Alert, Collapse, Typography } from 'antd';
import { CloudUploadOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { Panel } = Collapse;
const { Text } = Typography;

export const DatasetImportForm = ({ onImportStart }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const onFinish = async (values) => {
    setLoading(true);
    setStatus(null);
    try {
      const response = await fetch('/ingest/huggingface', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: values.dataset,
          subset: values.subset || undefined,
          text_column: values.textColumn,
          id_column: values.idColumn,
          split: values.split,
          batch_size: values.batchSize,
          workers: values.workers
        })
      });
      
      if (response.ok) {
        setStatus({ type: 'success', message: 'Dataset import started in background' });
        message.success('Dataset import started in background');
        form.resetFields();
        onImportStart?.();
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Import failed');
      }
    } catch (error) {
      setStatus({ type: 'error', message: error.message });
      message.error('Failed to start import: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="dataset-import-card" title={<><CloudUploadOutlined /> Import HuggingFace Dataset</>}>
      {status && (
        <Alert
          message={status.message}
          type={status.type}
          showIcon
          closable
          onClose={() => setStatus(null)}
          className="import-status-alert"
        />
      )}
      
      <Form form={form} onFinish={onFinish} layout="vertical" size="small">
        <Form.Item name="dataset" label="Dataset Path" rules={[{ required: true }]}>
          <Input placeholder="e.g., wikimedia/wikipedia" />
        </Form.Item>
        
        <Collapse ghost className="advanced-options">
          <Panel header="Advanced Options" key="1">
            <Form.Item name="subset" label="Subset/Config">
              <Input placeholder="e.g., 20231101.en (optional)" />
            </Form.Item>
            
            <Form.Item name="textColumn" label="Text Column" initialValue="text">
              <Input />
            </Form.Item>
            
            <Form.Item name="idColumn" label="ID Column" initialValue="id">
              <Input />
            </Form.Item>
            
            <Form.Item name="split" label="Split" initialValue="train">
              <Input />
            </Form.Item>
            
            <Form.Item name="batchSize" label="Batch Size" initialValue={1024}>
              <InputNumber min={1} max={10000} style={{ width: '100%' }} />
            </Form.Item>
            
            <Form.Item name="workers" label="Workers" initialValue={4}>
              <InputNumber min={1} max={16} style={{ width: '100%' }} />
            </Form.Item>
          </Panel>
        </Collapse>

        <Button 
          type="primary" 
          htmlType="submit" 
          loading={loading} 
          block
          icon={<CloudUploadOutlined />}
          className="import-btn"
        >
          Start Import
        </Button>
        
        <div className="import-help">
          <InfoCircleOutlined />
          <Text type="secondary">
            Import large datasets from HuggingFace Hub. This runs in the background.
          </Text>
        </div>
      </Form>
    </Card>
  );
};