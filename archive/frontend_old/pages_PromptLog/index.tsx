import { useState, useEffect } from 'react';
import { PromptLogTable } from '../../components/PromptLog/PromptLogTable';
import { PromptLogDashboard } from '../../components/PromptLog/PromptLogDashboard';
import { promptLogApi } from '../../services/promptLogApi';
import type { PromptTransaction, CostDashboardData } from '../../types/promptLog';
import { FileText, BarChart3, Database } from 'lucide-react';

export function PromptLogPage() {
  const [activeTab, setActiveTab] = useState<'log' | 'dashboard'>('log');
  
  const [transactions, setTransactions] = useState<PromptTransaction[]>([]);
  const [dashboardData, setDashboardData] = useState<CostDashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [txs, dashboard] = await Promise.all([
          promptLogApi.listTransactions(),
          promptLogApi.getDashboardData()
        ]);
        setTransactions(txs);
        setDashboardData(dashboard);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      padding: '40px 24px 60px',
      maxWidth: 1200,
      margin: '0 auto',
      width: '100%',
    }}>
      {/* Header */}
      <div style={{ marginBottom: 32, display: 'flex', alignItems: 'center', gap: 12 }}>
        <Database size={28} color="var(--color-accent)" />
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 800 }}>Nhật Ký Prompt</h1>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 15 }}>Quản lý chi phí và lịch sử gọi API LLM (Gemma, OpenAI, etc.)</p>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid var(--color-border)', paddingBottom: 12 }}>
        <button
          onClick={() => setActiveTab('log')}
          className={`btn ${activeTab === 'log' ? 'btn-primary' : 'btn-ghost'}`}
          style={{ padding: '8px 16px' }}
        >
          <FileText size={18} /> Nhật Ký Giao Dịch
        </button>
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`btn ${activeTab === 'dashboard' ? 'btn-primary' : 'btn-ghost'}`}
          style={{ padding: '8px 16px' }}
        >
          <BarChart3 size={18} /> Bảng Chi Phí
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px', color: 'var(--color-text-muted)' }}>
          Đang tải dữ liệu...
        </div>
      ) : (
        <>
          {activeTab === 'log' && <PromptLogTable transactions={transactions} />}
          {activeTab === 'dashboard' && dashboardData && <PromptLogDashboard data={dashboardData} />}
        </>
      )}
    </div>
  );
}
