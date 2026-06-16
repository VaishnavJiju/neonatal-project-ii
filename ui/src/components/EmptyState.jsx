import React from 'react';
import { AlertCircle, Search } from 'lucide-react';

const EmptyState = ({ message = "No data available for selected configuration", icon = "info" }) => {
    return (
        <div className="empty-state">
            <div className="empty-state-icon">
                {icon === "info" ? <AlertCircle size={60} /> : <Search size={60} />}
            </div>
            <h3 style={{ color: '#fff', fontSize: '1.2rem', marginBottom: '0.5rem' }}>Insight Unavailable</h3>
            <p style={{ maxWidth: '300px', lineHeight: 1.6 }}>{message}</p>
        </div>
    );
};

export default EmptyState;
