import { useAppContext } from '../context/AppContext';
import { Alert, Button, Card } from 'antd';
import { useState } from 'react';
import { LLMConfigModal } from './LLMConfigModal';
import { SettingOutlined } from '@ant-design/icons';

export const LlmProtected = ({ children }) => {
  const { llmConfig } = useAppContext();
  const [showConfig, setShowConfig] = useState(false);

  if (!llmConfig.configured) {
    return (
      <div className="llm-protected-container">
        <Card className="llm-protected-card">
          <Alert
            title="LLM Not Configured"
            description="Please configure your LLM settings before using chat."
            type="warning"
            showIcon
            className="llm-alert"
            action={
              <Button 
                type="primary" 
                onClick={() => setShowConfig(true)}
                icon={<SettingOutlined />}
                size="large"
              >
                Configure Now
              </Button>
            }
          />
        </Card>
        <LLMConfigModal 
          visible={showConfig} 
          onClose={() => setShowConfig(false)} 
        />
      </div>
    );
  }

  return children;
};
