import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import { AppProvider } from './context/AppContext';
import { NavBar } from './components/NavBar';
import { ChatPage } from './pages/ChatPage';
import { QueryPage } from './pages/QueryPage';
import { DashboardPage } from './pages/DashboardPage';
import { LlmProtected } from './components/LlmProtected';

const { Content } = Layout;

function App() {
  return (
    <AppProvider>
      <Router>
        <Layout className="app-layout">
          <NavBar />
          <Content className="app-content">
            <Routes>
              <Route path="/chat" element={
                <LlmProtected>
                  <ChatPage />
                </LlmProtected>
              } />
              <Route path="/query" element={<QueryPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/" element={
                <LlmProtected>
                  <ChatPage />
                </LlmProtected>
              } />
            </Routes>
          </Content>
        </Layout>
      </Router>
    </AppProvider>
  );
}

export default App;