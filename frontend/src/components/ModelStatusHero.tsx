import { ModelInfo } from '../api';

interface Props {
    info: ModelInfo | null;
    loading: boolean;
}

export function ModelStatusHero({ info, loading }: Props) {
    if (loading) return <div className="model-hero-loading">Caricamento AI...</div>;
    if (!info) return null;

    return (
        <div className="model-status-hero">
            <div className="status-items">
                {Object.entries(info.models).map(([name, model]) => (
                    <div key={name} className={`hero-status-item ${model.trained ? 'active' : 'inactive'}`}>
                        <span className="status-dot"></span>
                        <span className="model-name">{name}</span>
                    </div>
                ))}
            </div>
            <div className="hero-meta">
                <span>{info.total_samples} Spin totali</span>
                <span className="separator">•</span>
                <span>Training autom. ogni {info.retrain_interval}</span>
            </div>
        </div>
    );
}
