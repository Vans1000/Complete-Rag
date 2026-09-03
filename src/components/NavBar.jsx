import { useState } from 'react';
import { Layout, Menu, Button, Tag, Space, Tooltip, Badge } from 'antd';
import { SettingOutlined, DatabaseOutlined, MessageOutlined, SearchOutlined, SlidersOutlined } from '@ant-design/icons';
import { useAppContext } from '../context/AppContext';
import { LLMConfigModal } from './LLMConfigModal';
import { TokenizerConfigModal } from './TokenizerConfigModal';
import { Link, useLocation } from 'react-router-dom';

const { Header } = Layout;

export const NavBar = () => {
  const [configVisible, setConfigVisible] = useState(false);
  const [tokenizerVisible, setTokenizerVisible] = useState(false);
  const { llmConfig } = useAppContext();
  const location = useLocation();

  const items = [
    { 
      key: '/chat', 
      icon: <MessageOutlined />, 
      label: <Link to="/chat">Chat</Link> 
    },
    { 
      key: '/query', 
      icon: <SearchOutlined />, 
      label: <Link to="/query">Query</Link> 
    },
    { 
      key: '/dashboard', 
      icon: <DatabaseOutlined />, 
      label: <Link to="/dashboard">Dashboard</Link> 
    },
  ];

  return (
    <Header className="navbar">
      <div className="navbar-brand">
        <span className="brand-icon">🚀</span>
        <span className="brand-text">RAG Engine</span>
      </div>
      
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[location.pathname]}
        items={items}
        className="navbar-menu"
      />

      <Space className="navbar-actions">
        <Tooltip title={llmConfig.configured ? `Model: ${llmConfig.model}` : 'Click to configure LLM'}>
          <Tag 
            color={llmConfig.configured ? 'success' : 'error'} 
            className="llm-status-tag"
            onClick={() => setConfigVisible(true)}
          >
            {llmConfig.configured ? (
              <><Badge status="success" /> {llmConfig.model}</>
            ) : (
              <><Badge status="error" /> LLM Not Configured</>
            )}
          </Tag>
        </Tooltip>
        
        <Button 
          icon={<SlidersOutlined />} 
          onClick={() => setTokenizerVisible(true)}
        >
          Models
        </Button>

        <Button 
          icon={<SettingOutlined />} 
          onClick={() => setConfigVisible(true)}
          type="primary"
          className="config-btn"
        >
          Configure
        </Button>
      </Space>

      <LLMConfigModal
        visible={configVisible}
        onClose={() => setConfigVisible(false)}
      />

      <TokenizerConfigModal
        visible={tokenizerVisible}
        onClose={() => setTokenizerVisible(false)}
      />
    </Header>
  );
};