import { useState, useEffect } from 'react';
import { Select, Button, Modal, Input, Form, message, Space } from 'antd';

export const CollectionSelector = ({ value, onChange }) => {
  const [collections, setCollections] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchCollections();
    const interval = setInterval(fetchCollections, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchCollections = async () => {
    try {
      const res = await fetch('/collections');
      const data = await res.json();
      setCollections(data.collections);
      if (!value && data.collections.length > 0) {
        onChange(data.collections[0]);
      }
    } catch (error) {
      console.error('Failed to fetch collections');
    }
  };

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) return;
    setLoading(true);
    try {
      await fetch('/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newCollectionName })
      });
      message.success('Collection created');
      setIsModalOpen(false);
      fetchCollections();
      onChange(newCollectionName);
      setNewCollectionName('');
    } catch (error) {
      message.error('Failed to create collection');
    } finally {
      setLoading(false);
    }
  };

  const handleSwitch = async (collectionName) => {
    try {
      await fetch('/collections/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collection_name: collectionName })
      });
      onChange(collectionName);
      message.success(`Switched to ${collectionName}`);
    } catch (error) {
      message.error('Failed to switch collection');
    }
  };

  return (
    <Space className="collection-selector">
      <Select
        className="collection-select"
        placeholder="Select collection"
        value={value}
        onChange={handleSwitch}
        options={collections.map(c => ({ label: c, value: c }))}
      />
      <Button onClick={() => setIsModalOpen(true)} type="dashed">New</Button>
      
      <Modal
        title="Create New Collection"
        open={isModalOpen}
        onOk={handleCreateCollection}
        onCancel={() => setIsModalOpen(false)}
        confirmLoading={loading}
      >
        <Input
          placeholder="Collection name"
          value={newCollectionName}
          onChange={e => setNewCollectionName(e.target.value)}
          onPressEnter={handleCreateCollection}
        />
      </Modal>
    </Space>
  );
};
