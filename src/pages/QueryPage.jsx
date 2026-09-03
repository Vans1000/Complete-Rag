import { useState } from 'react';
import { Layout, Input, Button, Table, Card, Slider, Space, Typography, Upload, message, Badge, Spin } from 'antd';
import { SearchOutlined, InboxOutlined, FileOutlined, LoadingOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { CollectionSelector } from '../components/CollectionSelector';

const { Sider, Content } = Layout;
const { Title, Text } = Typography;

export const QueryPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(10);
  const [collection, setCollection] = useState(null);
  const [uploadingFiles, setUploadingFiles] = useState([]);

    const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('/api/query', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          top_k: topK,
          use_web_search: false
        })
      });
      
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      message.error('Query failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    const fileId = Date.now() + Math.random();
    const newFile = { id: fileId, name: file.name, status: 'uploading' };
    
    setUploadingFiles(prev => [...prev, newFile]);
    
    const formData = new FormData();
    formData.append('file', file);
    if (collection) formData.append('collection', collection);
    
    try {
      const response = await fetch('/ingest/file', { method: 'POST', body: formData });
      
      if (response.ok) {
        setUploadingFiles(prev => 
          prev.map(f => f.id === fileId ? { ...f, status: 'done' } : f)
        );
        message.success(`Uploaded ${file.name}`);
        setTimeout(() => {
          setUploadingFiles(prev => prev.filter(f => f.id !== fileId));
        }, 3000);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      setUploadingFiles(prev => 
        prev.map(f => f.id === fileId ? { ...f, status: 'error' } : f)
      );
      message.error('Upload failed');
    }
    return false;
  };

  const removeUploadingFile = (fileId) => {
    setUploadingFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const columns = [
    { 
      title: 'Score', 
      dataIndex: 'score', 
      key: 'score',
      sorter: (a, b) => a.score - b.score,
      render: score => (
        <span className={`score-badge ${score > 0.8 ? 'high' : score > 0.5 ? 'medium' : 'low'}`}>
          {score.toFixed(4)}
        </span>
      ),
      width: 100
    },
    { 
      title: 'Type', 
      dataIndex: 'type', 
      key: 'type', 
      width: 100,
      render: type => (
        <Badge 
          color={type === 'image' ? 'blue' : type === 'web' ? 'purple' : 'green'} 
          text={type || 'unknown'}
        />
      )
    },
    { 
      title: 'Source', 
      dataIndex: 'source', 
      key: 'source', 
      ellipsis: true,
      render: source => <Text ellipsis style={{ maxWidth: 200 }}>{source}</Text>
    },
    { 
      title: 'Page', 
      dataIndex: 'page', 
      key: 'page', 
      width: 80,
      render: page => page || '-'
    },
    { 
      title: 'Content', 
      dataIndex: 'content', 
      key: 'content',
      render: text => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {text?.length > 200 ? text.substring(0, 200) + '...' : text}
        </Text>
      )
    }
  ];

  return (
    <Layout className="query-layout">
      <Sider width={320} className="query-sider">
        <Card size="small" title="Collection" className="sider-card">
          <CollectionSelector value={collection} onChange={setCollection} />
        </Card>

        <Card size="small" title="Upload Documents" className="sider-card">
          <Upload.Dragger 
            beforeUpload={handleFileUpload} 
            showUploadList={false} 
            multiple
            className="upload-dragger"
          >
            <p className="upload-icon"><InboxOutlined /></p>
            <p className="upload-text">Drop docs here to upload</p>
          </Upload.Dragger>

          {uploadingFiles.length > 0 && (
            <div className="uploading-files">
              <Text type="secondary" className="uploading-title">Uploading:</Text>
              {uploadingFiles.map(file => (
                <div key={file.id} className="uploading-file-item">
                  <FileOutlined className="file-icon" />
                  <span className="file-name" title={file.name}>{file.name}</span>
                  {file.status === 'uploading' && (
                    <Spin indicator={<LoadingOutlined spin className="loading-icon" />} size="small" />
                  )}
                  {file.status === 'done' && <Badge status="success" text="Done" />}
                  {file.status === 'error' && <Badge status="error" text="Failed" />}
                  <Button 
                    type="text" 
                    size="small" 
                    icon={<CloseCircleOutlined />} 
                    onClick={() => removeUploadingFile(file.id)}
                    className="remove-btn"
                  />
                </div>
              ))}
            </div>
          )}
        </Card>
      </Sider>

      <Content className="query-content">
        <Title level={2} className="page-title">Vector Similarity Search</Title>
        
        <Card className="search-card">
          <Space orientation="vertical" style={{ width: '100%' }}>
            <Input.Search
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Enter query text..."
              enterButton={<><SearchOutlined /> Search</>}
              size="large"
              onSearch={() => handleSearch()} 
              loading={loading}
              className="search-input"
            />
            <div className="slider-container">
              <Text type="secondary">Top K Results: <strong>{topK}</strong></Text>
              <Slider min={1} max={50} value={topK} onChange={setTopK} className="topk-slider" />
            </div>
          </Space>
        </Card>

        <Card className="results-card" title={`Results (${results.length})`}>
          <Table 
            dataSource={results} 
            columns={columns} 
            rowKey="id"
            pagination={{ pageSize: 10 }}
            loading={loading}
            scroll={{ x: 'max-content' }}
            locale={{ emptyText: 'No results yet. Enter a query to search.' }}
          />
        </Card>
      </Content>
    </Layout>
  );
};
