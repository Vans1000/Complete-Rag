import { useState, useEffect, useCallback } from 'react';
import { Layout, Card, Statistic, Table, Tag, Button, Space, Typography, Tabs, List, Spin, Badge, Empty, Progress } from 'antd';
import { ReloadOutlined, DatabaseOutlined, FileOutlined, CloudUploadOutlined, LoadingOutlined, CheckCircleOutlined, CloseCircleOutlined, InboxOutlined } from '@ant-design/icons';
import { DatasetImportForm } from '../components/DatasetImportForm';

const { Content } = Layout;
const { Title, Text } = Typography;

export const DashboardPage = () => {
  const [stats, setStats] = useState(null);
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [activeUploads, setActiveUploads] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/dashboard/stats');
      const data = await res.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats');
    }
  }, []);

  const fetchPoints = useCallback(async (newOffset = null) => {
    setLoading(true);
    try {
      const res = await fetch('/dashboard/points', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 20, offset: newOffset })
      });
      const data = await res.json();
      if (newOffset) {
        setPoints(prev => [...prev, ...data.points]);
      } else {
        setPoints(data.points);
      }
      setOffset(data.next_offset);
    } catch (error) {
      console.error('Failed to fetch points');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    fetchPoints();
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchStats();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, fetchStats, fetchPoints]);

  const handleFileUpload = async (file) => {
    const uploadId = Date.now() + Math.random();
    const newUpload = { 
      id: uploadId, 
      name: file.name, 
      status: 'uploading',
      progress: 0 
    };
    
    setActiveUploads(prev => [...prev, newUpload]);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const progressInterval = setInterval(() => {
        setActiveUploads(prev => 
          prev.map(u => u.id === uploadId && u.status === 'uploading' 
            ? { ...u, progress: Math.min(u.progress + 10, 90) } 
            : u
          )
        );
      }, 300);

      const response = await fetch('/ingest/file', {
        method: 'POST',
        body: formData
      });
      
      clearInterval(progressInterval);
      
      if (response.ok) {
        setActiveUploads(prev => prev.filter(u => u.id !== uploadId));
        setUploadedFiles(prev => [...prev, { 
          name: file.name, 
          uploadedAt: new Date().toLocaleString(),
          status: 'processing'
        }]);
        
        setTimeout(() => {
          fetchStats();
          fetchPoints();
        }, 1000);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      setActiveUploads(prev => 
        prev.map(u => u.id === uploadId ? { ...u, status: 'error' } : u)
      );
    }
    return false;
  };

  const columns = [
    { 
      title: 'ID', 
      dataIndex: 'id', 
      key: 'id', 
      width: 100,
      render: id => <Text code copyable>{id.slice(0, 8)}...</Text>
    },
    { 
      title: 'Type', 
      key: 'type',
      render: (_, record) => (
        <Tag color={record.payload?.type === 'image' ? 'blue' : record.payload?.type === 'web' ? 'purple' : 'green'}>
          {record.payload?.type || 'unknown'}
        </Tag>
      ),
      width: 100
    },
    { 
      title: 'Source', 
      key: 'source',
      render: (_, record) => (
        <Text ellipsis style={{ maxWidth: 250 }}>
          {record.payload?.path || record.payload?.source || 'N/A'}
        </Text>
      ),
      ellipsis: true
    },
    { 
      title: 'Preview', 
      key: 'preview',
      render: (_, record) => {
        const text = record.payload?.text || record.payload?.caption || '';
        return (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {text.length > 100 ? text.substring(0, 100) + '...' : text}
          </Text>
        );
      }
    },
    { 
      title: 'Page', 
      key: 'page',
      render: (_, record) => record.payload?.page || '-',
      width: 80
    }
  ];

  const UploadProgress = () => (
    <div className="upload-progress-section">
      {activeUploads.map(upload => (
        <div key={upload.id} className="upload-item">
          <div className="upload-info">
            <FileOutlined className="upload-icon" />
            <Text ellipsis className="upload-name">{upload.name}</Text>
            {upload.status === 'uploading' && (
              <Spin indicator={<LoadingOutlined spin />} size="small" />
            )}
            {upload.status === 'error' && (
              <CloseCircleOutlined className="error-icon" />
            )}
          </div>
          {upload.status === 'uploading' && (
            <Progress percent={upload.progress} size="small" status="active" />
          )}
        </div>
      ))}
    </div>
  );

  const tabItems = [
    {
      key: 'documents',
      label: (
        <span>
          <DatabaseOutlined /> Stored Documents
          {stats && <Badge count={stats.vectors_count} style={{ marginLeft: 8 }} />}
        </span>
      ),
      children: (
        <>
          <Table 
            dataSource={points} 
            columns={columns} 
            rowKey="id"
            pagination={false}
            loading={loading}
            scroll={{ x: 'max-content' }}
          />
          {offset && (
            <Button 
              onClick={() => fetchPoints(offset)} 
              loading={loading}
              style={{ marginTop: 16 }}
              block
            >
              Load More
            </Button>
          )}
        </>
      )
    },
    {
      key: 'uploads',
      label: (
        <span>
          <CloudUploadOutlined /> File Uploads
          {(activeUploads.length + uploadedFiles.length) > 0 && (
            <Badge count={activeUploads.length + uploadedFiles.length} style={{ marginLeft: 8 }} />
          )}
        </span>
      ),
      children: (
        <div className="uploads-section">
          {/* Upload Area */}
          <Card className="upload-card" title="Upload New Files">
            <div 
              className="upload-dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const files = Array.from(e.dataTransfer.files);
                files.forEach(handleFileUpload);
              }}
            >
              <InboxOutlined className="dropzone-icon" />
              <Text className="dropzone-text">Drag & drop files here</Text>
              <Text type="secondary" className="dropzone-subtext">or click to browse</Text>
              <input 
                type="file" 
                multiple 
                onChange={(e) => Array.from(e.target.files).forEach(handleFileUpload)}
                className="file-input"
              />
            </div>
          </Card>

          {activeUploads.length > 0 && (
            <Card className="upload-card" title="Uploading...">
              <UploadProgress />
            </Card>
          )}

          {uploadedFiles.length > 0 && (
            <Card className="upload-card" title="Recent Uploads">
              <List
                dataSource={uploadedFiles.slice().reverse()}
                renderItem={file => (
                  <List.Item className="uploaded-file-item">
                    <div className="file-info">
                      <CheckCircleOutlined className="success-icon" />
                      <div className="file-details">
                        <Text strong className="file-name">{file.name}</Text>
                        <Text type="secondary" className="file-time">{file.uploadedAt}</Text>
                      </div>
                    </div>
                    <Tag color="processing">Processing</Tag>
                  </List.Item>
                )}
              />
            </Card>
          )}

          {activeUploads.length === 0 && uploadedFiles.length === 0 && (
            <Empty 
              description="No files uploaded yet" 
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </div>
      )
    },
    {
      key: 'huggingface',
      label: (
        <span>
          <CloudUploadOutlined /> HuggingFace Import
        </span>
      ),
      children: <DatasetImportForm onImportStart={fetchStats} />
    }
  ];

  return (
    <Content className="dashboard-content">
      <div className="dashboard-header">
        <Title level={2} className="dashboard-title">
          <DatabaseOutlined /> Qdrant Dashboard
        </Title>
        <Space>
          <Button 
            icon={<ReloadOutlined spin={loading} />} 
            onClick={() => { fetchStats(); fetchPoints(); }}
            loading={loading}
          >
            Refresh
          </Button>
        </Space>
      </div>

      {stats && (
        <Space size="large" className="stats-row">
          <Card className="stat-card">
            <Statistic 
              title="Collection" 
              value={stats.collection_name} 
              style={{ fontSize: 18 }}
            />
          </Card>
          <Card className="stat-card">
            <Statistic 
              title="Total Points" 
              value={stats.vectors_count} 
              style={{ color: '#3f8600', fontSize: 24 }}
            />
          </Card>
          <Card className="stat-card">
            <Statistic 
              title="Vector Dimensions" 
              value={512} 
              style={{ fontSize: 24 }}
            />
          </Card>
          <Card className="stat-card">
            <Statistic 
              title="Distance Metric" 
              value="Cosine" 
              style={{ fontSize: 18 }}
            />
          </Card>
        </Space>
      )}

      <Card className="dashboard-tabs-card">
        <Tabs items={tabItems} defaultActiveKey="documents" />
      </Card>
    </Content>
  );
};
