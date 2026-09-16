import { Radio, Checkbox, Space, Tooltip } from 'antd';
import { useAppContext } from '../context/AppContext';

export const ModeToggle = () => {
  const { mode, setMode, webIngest, setWebIngest } = useAppContext();
  const isWebEnabled = mode === 'web' || mode === 'force_web';

  return (
    <Space orientation="vertical" className="mode-toggle">
      <Radio.Group 
        value={mode} 
        onChange={e => setMode(e.target.value)}
        className="mode-radio-group"
      >
        <Tooltip title="Use only local documents">
          <Radio.Button value="local" className="mode-btn">
            <span className="mode-icon">📄 Local Only</span> 
          </Radio.Button>
        </Tooltip>
        <Tooltip title="Use local docs + web search">
          <Radio.Button value="web" className="mode-btn">
            <span className="mode-icon">🌐 Web</span> 
          </Radio.Button>
        </Tooltip>
        <Tooltip title="Prioritize web results">
          <Radio.Button value="force_web" className="mode-btn">
            <span className="mode-icon">⚡ Force Web</span> 
          </Radio.Button>
        </Tooltip>
      </Radio.Group>
      
      {isWebEnabled && (
        <Checkbox 
          checked={webIngest} 
          onChange={e => setWebIngest(e.target.checked)}
          className="web-ingest-checkbox"
        >
          Ingest web results into database
        </Checkbox>
      )}
    </Space>
  );
};
