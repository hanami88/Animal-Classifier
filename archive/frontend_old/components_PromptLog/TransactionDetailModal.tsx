import React from 'react';
import { X } from 'lucide-react';
import type { PromptTransaction } from '../../types/promptLog';

interface ModalProps {
  transaction: PromptTransaction | null;
  onClose: () => void;
}

export const TransactionDetailModal: React.FC<ModalProps> = ({ transaction, onClose }) => {
  if (!transaction) return null;

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)'
    }}>
      <div style={{
        background: 'var(--color-surface)', width: '90%', maxWidth: 800, maxHeight: '90vh',
        borderRadius: 16, boxShadow: '0 20px 40px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column',
        overflow: 'hidden', border: '1px solid var(--color-border)'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px 24px', borderBottom: '1px solid var(--color-border)' }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>Chi Tiết Giao Dịch</h2>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)' }}>
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: 24, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 32 }}>
          {/* Info Section */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 16 }}>Thông tin giao dịch</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>UUID</p>
                <p style={{ fontSize: 14, fontFamily: 'monospace', wordBreak: 'break-all' }}>{transaction.id}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Nhà cung cấp</p>
                <p style={{ fontSize: 14, fontWeight: 600 }}>{transaction.provider}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Model</p>
                <p style={{ fontSize: 14 }}>{transaction.model}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Tính năng</p>
                <p style={{ fontSize: 14 }}>{transaction.feature}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Phương thức</p>
                <p style={{ fontSize: 14 }}>{transaction.method}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Thời gian</p>
                <p style={{ fontSize: 14 }}>{new Date(transaction.timestamp).toLocaleString('vi-VN')}</p>
              </div>
            </div>
          </div>

          {/* Usage & Cost Section */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 16 }}>Sử dụng & Chi phí</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Token đầu vào</p>
                <p style={{ fontSize: 14 }}>{transaction.inputTokens.toLocaleString()}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Token đầu ra</p>
                <p style={{ fontSize: 14 }}>{transaction.outputTokens.toLocaleString()}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Tổng token</p>
                <p style={{ fontSize: 14 }}>{transaction.totalTokens.toLocaleString()}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Chi phí ước tính</p>
                <p style={{ fontSize: 14 }}>${transaction.cost.toFixed(6)}</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Độ trễ</p>
                <p style={{ fontSize: 14 }}>{transaction.latencyMs.toLocaleString()}ms</p>
              </div>
              <div>
                <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginBottom: 4 }}>Trạng thái</p>
                <span className={`badge badge-${transaction.status}`}>{transaction.status}</span>
              </div>
            </div>
          </div>

          {/* Prompt Section */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Prompt</h3>
            <div style={{ background: 'var(--color-surface-2)', padding: 16, borderRadius: 8, fontSize: 14, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {transaction.prompt}
            </div>
          </div>

          {/* System Instruction */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Chỉ dẫn hệ thống</h3>
            <div style={{ background: 'var(--color-surface-2)', padding: 16, borderRadius: 8, fontSize: 14, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {transaction.systemInstruction}
            </div>
          </div>

          {/* Response */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Phản hồi</h3>
            <div style={{ background: 'var(--color-surface-2)', padding: 16, borderRadius: 8, fontSize: 14, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {transaction.response}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
