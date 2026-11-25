import React, { useState } from 'react';
import { Search } from 'lucide-react';
import type { PromptTransaction } from '../../types/promptLog';
import { TransactionDetailModal } from './TransactionDetailModal';

interface TableProps {
  transactions: PromptTransaction[];
}

export const PromptLogTable: React.FC<TableProps> = ({ transactions }) => {
  const [selectedTx, setSelectedTx] = useState<PromptTransaction | null>(null);

  return (
    <div className="glass-card" style={{ padding: 24, overflowX: 'auto' }}>
      {/* Filters bar (Mock UI) */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        <input type="date" className="btn-ghost" style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--color-border)' }} />
        <select className="btn-ghost" style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--color-border)' }}>
          <option>Tất cả nhà cung cấp</option>
        </select>
        <select className="btn-ghost" style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--color-border)' }}>
          <option>Tất cả tính năng</option>
        </select>
        <select className="btn-ghost" style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid var(--color-border)' }}>
          <option>Tất cả trạng thái</option>
        </select>
        <button className="btn btn-primary" style={{ padding: '8px 16px' }}><Search size={16} /> Tìm kiếm</button>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--color-border)', color: 'var(--color-text-muted)', fontSize: 12, textTransform: 'uppercase' }}>
            <th style={{ padding: '12px 8px' }}>Thời gian</th>
            <th style={{ padding: '12px 8px' }}>Nhà cung cấp</th>
            <th style={{ padding: '12px 8px' }}>Model</th>
            <th style={{ padding: '12px 8px' }}>Tính năng</th>
            <th style={{ padding: '12px 8px' }}>Phương thức</th>
            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Token đầu vào</th>
            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Token đầu ra</th>
            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Chi phí</th>
            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Độ trễ</th>
            <th style={{ padding: '12px 8px', textAlign: 'center' }}>Trạng thái</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map(tx => (
            <tr 
              key={tx.id} 
              onClick={() => setSelectedTx(tx)}
              style={{ borderBottom: '1px solid var(--color-border)', cursor: 'pointer', transition: 'background 0.2s' }}
              className="hover:bg-slate-50 dark:hover:bg-slate-800/50"
            >
              <td style={{ padding: '16px 8px', fontSize: 13 }}>{new Date(tx.timestamp).toLocaleString('vi-VN')}</td>
              <td style={{ padding: '16px 8px', fontSize: 13 }}>{tx.provider}</td>
              <td style={{ padding: '16px 8px', fontSize: 13 }}>{tx.model}</td>
              <td style={{ padding: '16px 8px', fontSize: 13 }}>{tx.feature}</td>
              <td style={{ padding: '16px 8px', fontSize: 13 }}>{tx.method}</td>
              <td style={{ padding: '16px 8px', fontSize: 13, textAlign: 'right' }}>{tx.inputTokens.toLocaleString()}</td>
              <td style={{ padding: '16px 8px', fontSize: 13, textAlign: 'right' }}>{tx.outputTokens.toLocaleString()}</td>
              <td style={{ padding: '16px 8px', fontSize: 13, textAlign: 'right' }}>${tx.cost.toFixed(6)}</td>
              <td style={{ padding: '16px 8px', fontSize: 13, textAlign: 'right' }}>{tx.latencyMs.toLocaleString()}ms</td>
              <td style={{ padding: '16px 8px', textAlign: 'center' }}>
                <span className={`badge badge-${tx.status}`}>{tx.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {selectedTx && (
        <TransactionDetailModal transaction={selectedTx} onClose={() => setSelectedTx(null)} />
      )}
    </div>
  );
};
