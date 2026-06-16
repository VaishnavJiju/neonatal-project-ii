import React, { createContext, useState, useContext } from 'react';

const NexusContext = createContext();

export const NexusProvider = ({ children }) => {
    const [globalTarget, setGlobalTarget] = useState('');
    const [globalFeatures, setGlobalFeatures] = useState([]);
    const [trainedMetrics, setTrainedMetrics] = useState(null);
    const [datasetVariant, setDatasetVariant] = useState('daily');
    const [taskType, setTaskType] = useState('regression');
    const [colsList, setColsList] = useState([]); // Needed globally now
    const [modelVersion, setModelVersion] = useState('v1'); // v1 (Clinical) or v0 (Full)

    return (
        <NexusContext.Provider value={{
            globalTarget, setGlobalTarget,
            globalFeatures, setGlobalFeatures,
            trainedMetrics, setTrainedMetrics,
            datasetVariant, setDatasetVariant,
            taskType, setTaskType,
            colsList, setColsList,
            modelVersion, setModelVersion
        }}>
            {children}
        </NexusContext.Provider>
    );
};

export const useNexus = () => useContext(NexusContext);
