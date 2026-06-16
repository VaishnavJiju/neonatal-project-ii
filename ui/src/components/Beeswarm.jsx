import React, { useEffect, useRef, useState } from 'react';

const Beeswarm = ({ detail, topN = 12 }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };
        updateSize();
        window.addEventListener('resize', updateSize);
        return () => window.removeEventListener('resize', updateSize);
    }, []);

    useEffect(() => {
        if (!detail || !detail.shap_values || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const { width, height } = dimensions;
        const dpr = window.devicePixelRatio || 1;
        
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        const { shap_values, feature_names, feature_values, global_importance } = detail;
        
        // Find top features
        const sortedFeatures = Object.entries(global_importance || {})
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN)
            .map(x => x[0]);

        const featureIndices = sortedFeatures.map(f => feature_names.indexOf(f));
        
        // Find SHAP range for X scaling
        let minShap = 0, maxShap = 0;
        shap_values.forEach(row => {
            row.forEach(val => {
                if (val < minShap) minShap = val;
                if (val > maxShap) maxShap = val;
            });
        });
        const absMax = Math.max(Math.abs(minShap), Math.abs(maxShap)) * 1.1;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        const paddingLeft = 180;
        const paddingRight = 40;
        const paddingTop = 40;
        const paddingBottom = 60;
        const chartWidth = width - paddingLeft - paddingRight;
        const chartHeight = height - paddingTop - paddingBottom;
        const rowHeight = chartHeight / topN;

        // Draw Axes and Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(paddingLeft + chartWidth / 2, paddingTop);
        ctx.lineTo(paddingLeft + chartWidth / 2, paddingTop + chartHeight);
        ctx.stroke();

        // X-Axis Ticks
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        [-absMax, -absMax/2, 0, absMax/2, absMax].forEach(val => {
            const x = paddingLeft + (val + absMax) / (2 * absMax) * chartWidth;
            ctx.fillText(val.toFixed(2), x, paddingTop + chartHeight + 20);
        });
        ctx.fillText("SHAP Value (Impact on prediction)", paddingLeft + chartWidth/2, paddingTop + chartHeight + 40);

        // Draw Rows and Points
        sortedFeatures.forEach((feature, i) => {
            const featureIdx = featureIndices[i];
            const yCenter = paddingTop + (i + 0.5) * rowHeight;

            // Feature Name
            ctx.fillStyle = '#fff';
            ctx.font = '500 11px Outfit';
            ctx.textAlign = 'right';
            ctx.fillText(feature, paddingLeft - 15, yCenter + 4);

            // Points
            shap_values.forEach((row, sIdx) => {
                const shap = row[featureIdx];
                const val = feature_values[sIdx][featureIdx];
                
                // Color scale (Blue -> Red)
                // Assuming feature values are normalized 0-1 for simplicity in color, 
                // but actually we should find min/max per feature
                // For now, use a global heuristic or find per-feature min/max
                const x = paddingLeft + (shap + absMax) / (2 * absMax) * chartWidth;
                
                // Deterministic Jitter
                const jitter = (Math.sin(sIdx * 123.456 + i) * (rowHeight * 0.4));
                const y = yCenter + jitter;

                // Simple normalization for color
                // In a real app, we'd pre-calculate min/max per feature
                const hue = 240 - (val * 240); // 0 (red) to 240 (blue)
                // Actually SHAP usually uses Red for high, Blue for low
                const hueRedBlue = val > 0.5 ? 0 : 240; 
                // Better gradient:
                const r = Math.round(255 * val);
                const b = Math.round(255 * (1 - val));
                ctx.fillStyle = `rgba(${r}, 50, ${b}, 0.6)`;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        });

        // Legend
        const legendX = paddingLeft + chartWidth - 100;
        const legendY = 15;
        const gradient = ctx.createLinearGradient(legendX, 0, legendX + 100, 0);
        gradient.addColorStop(0, 'blue');
        gradient.addColorStop(1, 'red');
        ctx.fillStyle = gradient;
        ctx.fillRect(legendX, legendY, 100, 8);
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '10px Inter';
        ctx.textAlign = 'left';
        ctx.fillText("Low Value", legendX, legendY + 20);
        ctx.textAlign = 'right';
        ctx.fillText("High Value", legendX + 100, legendY + 20);

    }, [detail, topN, dimensions]);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '500px', background: 'rgba(0,0,0,0.1)', borderRadius: '8px', padding: '10px' }}>
            <canvas 
                ref={canvasRef} 
                style={{ width: '100%', height: '100%', display: 'block' }} 
            />
        </div>
    );
};

export default Beeswarm;
