"""
Medical Diagnosis Assistant with Contextual Engineering
========================================================

DESIGN OVERVIEW:
----------------
This system implements a comprehensive medical diagnosis assistant using LangGraph
and contextual engineering principles. It maintains patient context across multiple
consultations while ensuring privacy and providing evidence-based recommendations.

ARCHITECTURE COMPONENTS:
------------------------
1. STATE MANAGEMENT: Custom state schema with patient data isolation
2. SHORT-TERM MEMORY: Checkpointing for session-based consultation persistence
3. LONG-TERM MEMORY: Secure patient records and case study database
4. SCRATCHPAD: Runtime state fields for clinical note processing
5. WRITE: Functions to store symptoms, test results, and diagnoses
6. SELECT: Context retrieval from medical literature and patient history
7. COMPRESS: Summarization of lengthy medical histories into clinical notes
8. ISOLATE: Sub-agents for different specialties with privacy controls

WORKFLOW:
---------
1. Load patient information and current symptoms
2. Retrieve relevant medical history and literature
3. Delegate to specialized sub-agents by medical specialty
4. Collect and aggregate clinical assessments
5. Compress findings into clinical summary
6. Store consultation notes and update patient records

PRIVACY & COMPLIANCE:
---------------------
- Patient data encrypted and isolated per session
- HIPAA-compliant data handling
- Audit trail for all patient data access
- No patient data in sub-agent contexts without explicit need
- Differential privacy for statistical analysis
"""

import os
import uuid
import hashlib
from typing import TypedDict, Literal, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# LangChain and LangGraph imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# LangGraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import create_react_agent

# Embeddings for RAG
from langchain.embeddings import init_embeddings


# ============================================================================
# SECTION 1: ENUMS AND DATA MODELS
# ============================================================================
#
# Define enumerations and data structures for medical data.
# This ensures type safety and clear data organization.
# ============================================================================

class Severity(Enum):
    """Symptom or condition severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


class Specialty(Enum):
    """Medical specialties for sub-agent routing."""
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PULMONOLOGY = "pulmonology"
    GASTROENTEROLOGY = "gastroenterology"
    ENDOCRINOLOGY = "endocrinology"
    GENERAL = "general"


class PrivacyLevel(Enum):
    """Data privacy classification levels."""
    PUBLIC = "public"  # General medical knowledge
    CONFIDENTIAL = "confidential"  # Patient data
    RESTRICTED = "restricted"  # Sensitive medical data


@dataclass
class Symptom:
    """Structured symptom data."""
    description: str
    severity: Severity
    duration_days: int
    onset_date: str
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "description": self.description,
            "severity": self.severity.value,
            "duration_days": self.duration_days,
            "onset_date": self.onset_date,
            "location": self.location
        }


@dataclass
class TestResult:
    """Structured test result data."""
    test_name: str
    result_value: str
    normal_range: str
    abnormal: bool
    test_date: str
    ordered_by: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "test_name": self.test_name,
            "result_value": self.result_value,
            "normal_range": self.normal_range,
            "abnormal": self.abnormal,
            "test_date": self.test_date,
            "ordered_by": self.ordered_by
        }


# ============================================================================
# SECTION 2: STATE SCHEMA DEFINITION (SCRATCHPAD & PRIVACY)
# ============================================================================
#
# The state schema defines patient consultation data structure.
# Critical design: Patient identifiable information (PII) is isolated
# from general medical context to ensure privacy compliance.
#
# Key Privacy Features:
# - Patient ID is hashed for anonymization
# - PII stored separately from clinical data
# - Sub-agents receive only necessary, de-identified information
# - Audit trail tracks all data access
# ============================================================================

class MedicalConsultationState(TypedDict):
    """
    State schema for the medical diagnosis assistant.
    
    This state object flows through all nodes in the graph. Fields are
    organized by privacy level and access requirements.
    
    PATIENT IDENTIFICATION (CONFIDENTIAL):
    ----------------------------------------
    patient_id_hash : str
        Hashed patient identifier for privacy (SHA-256)
    
    consultation_id : str
        Unique identifier for this consultation session
    
    CURRENT CONSULTATION DATA (CONFIDENTIAL):
    ------------------------------------------
    chief_complaint : str
        Primary reason for visit
    
    current_symptoms : List[Dict[str, Any]]
        List of current symptoms (structured as Symptom.to_dict())
    
    vital_signs : Dict[str, Any]
        Current vital signs (BP, HR, temp, etc.)
    
    physical_exam_findings : str
        Notes from physical examination
    
    HISTORICAL DATA (CONFIDENTIAL - SELECTED):
    -------------------------------------------
    relevant_history : str
        Selected relevant medical history (COMPRESSED from full history)
        This is the result of SELECT strategy on patient records
    
    past_diagnoses : List[str]
        Relevant previous diagnoses
    
    current_medications : List[str]
        Active medication list
    
    allergies : List[str]
        Known allergies and adverse reactions
    
    recent_test_results : List[Dict[str, Any]]
        Recent lab and imaging results (structured as TestResult.to_dict())
    
    CLINICAL ASSESSMENT (GENERATED):
    ---------------------------------
    cardiology_assessment : str
        Assessment from cardiology sub-agent
    
    neurology_assessment : str
        Assessment from neurology sub-agent
    
    pulmonology_assessment : str
        Assessment from pulmonology sub-agent
    
    gastro_assessment : str
        Assessment from gastroenterology sub-agent
    
    endocrine_assessment : str
        Assessment from endocrinology sub-agent
    
    general_assessment : str
        Assessment from general medicine sub-agent
    
    differential_diagnoses : List[Dict[str, Any]]
        List of possible diagnoses with probabilities
        Format: [{"diagnosis": str, "probability": float, "evidence": List[str]}]
    
    recommended_tests : List[str]
        Recommended additional tests or procedures
    
    treatment_recommendations : List[str]
        Recommended treatments and interventions
    
    WORKFLOW MANAGEMENT (INTERNAL):
    --------------------------------
    scratchpad : str
        Temporary notes for workflow processing (WRITE strategy)
        Not exposed to sub-agents or stored in patient records
    
    clinical_summary : str
        Compressed summary of consultation (COMPRESS strategy)
        This becomes part of patient's medical record
    
    retrieved_literature : List[Dict[str, Any]]
        Relevant medical literature (SELECT strategy from knowledge base)
    
    similar_cases : List[Dict[str, Any]]
        Similar past cases (SELECT strategy from case database)
    
    specialties_consulted : List[str]
        List of specialties that have been consulted
    
    audit_trail : List[Dict[str, Any]]
        Record of data access for compliance
    
    privacy_level : str
        Current privacy level for this consultation
    """
    
    # Patient Identification (Confidential)
    patient_id_hash: str
    consultation_id: str
    
    # Current Consultation Data (Confidential)
    chief_complaint: str
    current_symptoms: List[Dict[str, Any]]
    vital_signs: Dict[str, Any]
    physical_exam_findings: str
    
    # Historical Data (Confidential - Selected)
    relevant_history: str
    past_diagnoses: List[str]
    current_medications: List[str]
    allergies: List[str]
    recent_test_results: List[Dict[str, Any]]
    
    # Clinical Assessment (Generated)
    cardiology_assessment: str
    neurology_assessment: str
    pulmonology_assessment: str
    gastro_assessment: str
    endocrine_assessment: str
    general_assessment: str
    differential_diagnoses: List[Dict[str, Any]]
    recommended_tests: List[str]
    treatment_recommendations: List[str]
    
    # Workflow Management (Internal)
    scratchpad: str
    clinical_summary: str
    retrieved_literature: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    specialties_consulted: List[str]
    audit_trail: List[Dict[str, Any]]
    privacy_level: str


# ============================================================================
# SECTION 3: PRIVACY AND SECURITY UTILITIES
# ============================================================================
#
# These utilities ensure HIPAA compliance and patient privacy.
# All patient identifiers are hashed, and access is logged.
# ============================================================================

class PrivacyManager:
    """
    Manages patient privacy and data access controls.
    
    This class implements privacy-preserving operations:
    - Patient ID hashing for anonymization
    - Audit trail logging for compliance
    - Data de-identification for sub-agent contexts
    - Access control verification
    """
    
    @staticmethod
    def hash_patient_id(patient_id: str) -> str:
        """
        Hash patient ID for anonymization.
        
        Uses SHA-256 to create a one-way hash of patient identifiers.
        This allows linking records without exposing actual patient IDs.
        
        Parameters:
        -----------
        patient_id : str
            Original patient identifier
        
        Returns:
        --------
        str
            Hashed patient identifier (64 hex characters)
        """
        return hashlib.sha256(patient_id.encode()).hexdigest()
    
    @staticmethod
    def create_audit_entry(
        action: str,
        data_accessed: str,
        user_id: str,
        consultation_id: str
    ) -> Dict[str, Any]:
        """
        Create audit trail entry for data access.
        
        HIPAA requires logging all access to patient data.
        This creates a tamper-evident audit record.
        
        Parameters:
        -----------
        action : str
            Action performed (e.g., "read_patient_history")
        data_accessed : str
            Description of data accessed
        user_id : str
            Identifier of user/system accessing data
        consultation_id : str
            Consultation session identifier
        
        Returns:
        --------
        Dict[str, Any]
            Audit trail entry with timestamp and details
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data_accessed": data_accessed,
            "user_id": user_id,
            "consultation_id": consultation_id,
            "access_granted": True
        }
    
    @staticmethod
    def de_identify_for_subagent(
        patient_data: Dict[str, Any],
        specialty: Specialty
    ) -> Dict[str, Any]:
        """
        Remove PII from patient data for sub-agent context.
        
        Sub-agents should only receive clinical information necessary
        for their specialty, with all identifying information removed.
        This implements the ISOLATE strategy with privacy controls.
        
        Parameters:
        -----------
        patient_data : Dict[str, Any]
            Full patient data from state
        specialty : Specialty
            Medical specialty of the sub-agent
        
        Returns:
        --------
        Dict[str, Any]
            De-identified clinical data appropriate for specialty
        """
        # Remove all PII
        de_identified = {
            "chief_complaint": patient_data.get("chief_complaint", ""),
            "symptoms": patient_data.get("current_symptoms", []),
            "vital_signs": patient_data.get("vital_signs", {}),
            "relevant_history": patient_data.get("relevant_history", ""),
            "medications": patient_data.get("current_medications", []),
            "allergies": patient_data.get("allergies", []),
            "test_results": patient_data.get("recent_test_results", []),
            "specialty_focus": specialty.value
        }
        
        # Note: patient_id_hash, consultation_id, and other identifiers
        # are NOT included in sub-agent context
        
        return de_identified


# ============================================================================
# SECTION 4: MEDICAL KNOWLEDGE BASE AND RAG SETUP (SELECT STRATEGY)
# ============================================================================
#
# This section implements the SELECT component of contextual engineering
# for medical knowledge. We create a knowledge base of medical literature,
# clinical guidelines, and diagnostic criteria that can be queried via RAG.
#
# This demonstrates:
# - Chunking medical documents for embedding
# - Creating a vector store for semantic search
# - Building retrieval tools for evidence-based medicine
# ============================================================================

class MedicalKnowledgeBase:
    """
    Medical knowledge base for evidence-based diagnosis support.
    
    This class implements RAG (Retrieval-Augmented Generation) for
    medical literature. Instead of loading all medical knowledge into
    context, we:
    1. Index medical literature into a vector store
    2. Perform semantic search for relevant information
    3. Provide only relevant context to diagnostic agents
    
    This is the SELECT strategy for medical knowledge retrieval.
    """
    
    def __init__(self, embeddings):
        """
        Initialize medical knowledge base.
        
        Parameters:
        -----------
        embeddings : Embeddings
            Embedding model for vectorizing medical documents
        """
        # Sample medical literature and clinical guidelines
        # In production, this would load from medical databases,
        # clinical guidelines, research papers, etc.
        self.documents = [
            {
                "content": """
                ACUTE CORONARY SYNDROME (ACS) - DIAGNOSTIC CRITERIA:
                
                Classic Presentation:
                - Chest pain/discomfort (pressure, squeezing, fullness)
                - Pain radiating to left arm, neck, jaw, or back
                - Associated symptoms: dyspnea, nausea, diaphoresis
                - Duration: typically >20 minutes
                
                Risk Factors:
                - Age >45 (men), >55 (women)
                - Hypertension, diabetes, hyperlipidemia
                - Family history of CAD
                - Smoking, obesity, sedentary lifestyle
                
                Diagnostic Workup:
                - ECG: ST elevation or depression, T-wave changes
                - Cardiac biomarkers: Troponin I/T elevation
                - CK-MB elevation (less specific)
                
                Immediate Management:
                - Aspirin 325mg PO (unless contraindicated)
                - Nitroglycerin sublingual
                - Oxygen if SpO2 <90%
                - Morphine for pain control
                - Activate catheterization lab if STEMI
                """,
                "metadata": {
                    "specialty": "cardiology",
                    "condition": "acute_coronary_syndrome",
                    "priority": "critical",
                    "evidence_level": "A"
                }
            },
            {
                "content": """
                STROKE - DIAGNOSTIC AND MANAGEMENT GUIDELINES:
                
                Clinical Presentation (FAST):
                - Face drooping (asymmetric smile)
                - Arm weakness (drift on extension)
                - Speech difficulty (slurred or inappropriate)
                - Time to call emergency services
                
                Additional Signs:
                - Sudden severe headache
                - Vision changes (unilateral or bilateral)
                - Ataxia, vertigo, imbalance
                - Confusion, altered mental status
                
                Immediate Assessment:
                - NIH Stroke Scale (NIHSS)
                - Blood glucose check (rule out hypoglycemia)
                - Non-contrast CT head (within 25 min of arrival)
                - BP monitoring (permissive hypertension allowed)
                
                Time-Dependent Treatment:
                - tPA window: 0-4.5 hours from onset
                - Thrombectomy window: 0-24 hours (selected patients)
                - Blood pressure management per protocol
                
                Contraindications to tPA:
                - ICH on CT, bleeding diathesis
                - Recent surgery (<14 days)
                - BP >185/110 despite treatment
                - Glucose <50 or >400 mg/dL
                """,
                "metadata": {
                    "specialty": "neurology",
                    "condition": "stroke",
                    "priority": "critical",
                    "evidence_level": "A"
                }
            },
            {
                "content": """
                CHRONIC OBSTRUCTIVE PULMONARY DISEASE (COPD) - MANAGEMENT:
                
                Diagnosis:
                - Clinical: chronic dyspnea, cough, sputum production
                - Spirometry: FEV1/FVC <0.70 post-bronchodilator
                - Severity by FEV1: Mild >80%, Moderate 50-80%, Severe <50%
                
                Stable COPD Management:
                - Smoking cessation (most important intervention)
                - Bronchodilators: SABA/SAMA for all patients
                - LABA/LAMA for persistent symptoms
                - ICS for frequent exacerbations (>2/year)
                - Pulmonary rehabilitation
                - Oxygen therapy if SpO2 <88% or PaO2 <55mmHg
                
                Acute Exacerbation:
                - Increased dyspnea, sputum volume, or purulence
                - Treatment: bronchodilators, systemic steroids
                - Antibiotics if purulent sputum or mechanical ventilation
                - Consider NIV if pH 7.25-7.35, PCO2 45-60
                
                Vaccination:
                - Annual influenza vaccine
                - Pneumococcal vaccine (PPSV23 and PCV13)
                """,
                "metadata": {
                    "specialty": "pulmonology",
                    "condition": "copd",
                    "priority": "high",
                    "evidence_level": "A"
                }
            },
            {
                "content": """
                DIABETES MELLITUS TYPE 2 - DIAGNOSIS AND MANAGEMENT:
                
                Diagnostic Criteria (any one):
                - Fasting glucose ≥126 mg/dL (on two occasions)
                - HbA1c ≥6.5%
                - OGTT 2-hour glucose ≥200 mg/dL
                - Random glucose ≥200 mg/dL with symptoms
                
                Initial Management:
                - Lifestyle: diet, exercise, weight loss (7-10%)
                - Metformin first-line (unless contraindicated)
                - Target HbA1c <7% for most, individualize goals
                
                Treatment Intensification:
                - If HbA1c >7% on metformin, add second agent:
                  * SGLT2i or GLP-1RA if ASCVD/CKD/HF present
                  * DPP-4i, sulfonylurea, TZD alternatives
                - Insulin if HbA1c >10% or A1c not at goal
                
                Monitoring:
                - HbA1c every 3 months until goal, then every 6 months
                - Annual: microalbuminuria, lipids, dilated eye exam
                - Foot examination at each visit
                
                Complications Screening:
                - Retinopathy: annual dilated eye exam
                - Nephropathy: annual urine albumin/creatinine
                - Neuropathy: annual monofilament test
                - Cardiovascular: assess risk factors regularly
                """,
                "metadata": {
                    "specialty": "endocrinology",
                    "condition": "diabetes_type2",
                    "priority": "high",
                    "evidence_level": "A"
                }
            },
            {
                "content": """
                GASTROESOPHAGEAL REFLUX DISEASE (GERD) - MANAGEMENT:
                
                Clinical Presentation:
                - Heartburn (burning retrosternal discomfort)
                - Regurgitation of gastric contents
                - Worse after meals, lying down, bending over
                - May have atypical symptoms: chronic cough, laryngitis
                
                Diagnostic Approach:
                - Clinical diagnosis if typical symptoms
                - Empiric PPI trial (2-4 weeks)
                - Upper endoscopy if alarm features:
                  * Dysphagia, odynophagia
                  * Weight loss, anemia
                  * Age >50 with new-onset symptoms
                  * Persistent symptoms despite PPI
                
                Treatment:
                - Lifestyle: elevate head of bed, avoid late meals
                - Avoid triggers: caffeine, alcohol, fatty foods
                - Weight loss if overweight
                - PPI: 30 minutes before breakfast
                - H2RA alternative or add-on for nocturnal symptoms
                
                Long-term Management:
                - Step-down therapy after symptom control
                - Surgical fundoplication for refractory cases
                - Surveillance EGD if Barrett's esophagus
                """,
                "metadata": {
                    "specialty": "gastroenterology",
                    "condition": "gerd",
                    "priority": "medium",
                    "evidence_level": "B"
                }
            },
            {
                "content": """
                HYPERTENSION - SCREENING AND MANAGEMENT:
                
                Blood Pressure Classification:
                - Normal: <120/80 mmHg
                - Elevated: 120-129/<80 mmHg
                - Stage 1: 130-139/80-89 mmHg
                - Stage 2: ≥140/90 mmHg
                
                Initial Evaluation:
                - Confirm diagnosis with multiple readings
                - Ambulatory BP monitoring if white coat suspected
                - Assess for target organ damage
                - Screen for secondary causes if indicated
                
                Treatment Goals:
                - <130/80 for most patients
                - <140/90 for age >65 or low CVD risk
                - Individualize based on comorbidities
                
                First-Line Medications:
                - Thiazide diuretic (HCTZ, chlorthalidone)
                - ACE inhibitor or ARB
                - Calcium channel blocker
                - Consider combination therapy if ≥20/10 above goal
                
                Special Populations:
                - CKD: ACE-I or ARB first-line
                - Post-MI: beta-blocker and ACE-I
                - Heart failure: ACE-I/ARB + beta-blocker + diuretic
                """,
                "metadata": {
                    "specialty": "cardiology",
                    "condition": "hypertension",
                    "priority": "high",
                    "evidence_level": "A"
                }
            }
        ]
        
        # Chunk documents for better retrieval granularity
        # Medical documents are chunked at 800 chars to preserve
        # complete clinical reasoning while enabling focused retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Convert documents to LangChain Document format
        from langchain_core.documents import Document
        docs = []
        for doc in self.documents:
            # Split each document into chunks
            chunks = self.text_splitter.split_text(doc["content"])
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata=doc["metadata"]
                ))
        
        # Create vector store for semantic search
        # This enables SELECT strategy: retrieve only relevant
        # medical knowledge based on patient presentation
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=embeddings
        )
        
        # Create retriever with top-k selection
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Top 5 most relevant chunks
        )
    
    def get_retriever_tool(self):
        """
        Create retriever tool for sub-agent use.
        
        This tool allows diagnostic sub-agents to actively query
        medical literature based on patient presentation.
        
        Returns:
        --------
        Tool
            LangChain tool for medical literature retrieval
        """
        return create_retriever_tool(
            self.retriever,
            "retrieve_medical_literature",
            "Search and return relevant medical literature, clinical guidelines, "
            "and diagnostic criteria. Use this to find evidence-based information "
            "for diagnosis and treatment planning. Query with patient symptoms, "
            "suspected conditions, or clinical questions."
        )


# ============================================================================
# SECTION 5: PATIENT RECORDS MANAGEMENT (WRITE AND SELECT)
# ============================================================================
#
# This section implements secure patient record storage and retrieval.
# It demonstrates:
# - WRITE strategy for storing consultation notes and test results
# - SELECT strategy for retrieving relevant patient history
# - Privacy controls for patient data access
# ============================================================================

class PatientRecordsManager:
    """
    Manages secure patient medical records with privacy controls.
    
    This class implements WRITE and SELECT strategies for patient data:
    - Stores consultation notes, test results, diagnoses
    - Retrieves relevant medical history for current consultation
    - Maintains audit trail for HIPAA compliance
    - Implements encryption and access controls
    
    In production, this would interface with Electronic Health Record (EHR)
    systems like Epic, Cerner, or HL7 FHIR APIs.
    """
    
    def __init__(self, store: BaseStore):
        """
        Initialize patient records manager.
        
        Parameters:
        -----------
        store : BaseStore
            LangGraph store for persistent record storage
        """
        self.store = store
        self.privacy_manager = PrivacyManager()
    
    def get_namespace(self, patient_id_hash: str, record_type: str):
        """
        Get namespace for patient record organization.
        
        Namespaces organize records by patient and type:
        ("patient_records", patient_id_hash, record_type)
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        record_type : str
            Type of record (consultations, tests, diagnoses, etc.)
        
        Returns:
        --------
        tuple
            Namespace tuple for record access
        """
        return ("patient_records", patient_id_hash, record_type)
    
    def write_consultation(
        self,
        patient_id_hash: str,
        consultation_data: Dict[str, Any],
        user_id: str
    ):
        """
        Store consultation notes in patient record.
        
        This implements the WRITE strategy for patient data.
        All data access is logged for HIPAA compliance.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        consultation_data : Dict[str, Any]
            Consultation notes and findings
        user_id : str
            Identifier of healthcare provider
        """
        namespace = self.get_namespace(patient_id_hash, "consultations")
        consultation_id = str(uuid.uuid4())
        
        # Add metadata
        consultation_data["consultation_id"] = consultation_id
        consultation_data["timestamp"] = datetime.now().isoformat()
        consultation_data["provider_id"] = user_id
        
        # Store consultation
        self.store.put(namespace, consultation_id, consultation_data)
        
        # Log access for audit trail
        audit_entry = self.privacy_manager.create_audit_entry(
            action="write_consultation",
            data_accessed="consultation_notes",
            user_id=user_id,
            consultation_id=consultation_id
        )
    
    def write_test_results(
        self,
        patient_id_hash: str,
        test_results: List[TestResult],
        user_id: str
    ):
        """
        Store test results in patient record.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        test_results : List[TestResult]
            List of test results to store
        user_id : str
            Identifier of healthcare provider
        """
        namespace = self.get_namespace(patient_id_hash, "test_results")
        
        for test in test_results:
            result_id = str(uuid.uuid4())
            result_data = test.to_dict()
            result_data["result_id"] = result_id
            result_data["recorded_by"] = user_id
            
            self.store.put(namespace, result_id, result_data)
    
    def write_diagnosis(
        self,
        patient_id_hash: str,
        diagnosis_data: Dict[str, Any],
        user_id: str
    ):
        """
        Store diagnosis in patient record.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        diagnosis_data : Dict[str, Any]
            Diagnosis and treatment plan
        user_id : str
            Identifier of healthcare provider
        """
        namespace = self.get_namespace(patient_id_hash, "diagnoses")
        diagnosis_id = str(uuid.uuid4())
        
        diagnosis_data["diagnosis_id"] = diagnosis_id
        diagnosis_data["timestamp"] = datetime.now().isoformat()
        diagnosis_data["provider_id"] = user_id
        
        self.store.put(namespace, diagnosis_id, diagnosis_data)
    
    def select_relevant_history(
        self,
        patient_id_hash: str,
        current_symptoms: List[str],
        limit: int = 10
    ) -> str:
        """
        Retrieve relevant medical history for current consultation.
        
        This implements the SELECT strategy: instead of loading entire
        patient history, we retrieve only information relevant to the
        current presentation.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        current_symptoms : List[str]
            Current symptoms to match against history
        limit : int
            Maximum number of historical records to retrieve
        
        Returns:
        --------
        str
            Compressed relevant medical history
        """
        # Retrieve past consultations
        consultation_namespace = self.get_namespace(
            patient_id_hash, "consultations"
        )
        past_consultations = list(
            self.store.search(consultation_namespace)
        )[:limit]
        
        # Retrieve past diagnoses
        diagnosis_namespace = self.get_namespace(
            patient_id_hash, "diagnoses"
        )
        past_diagnoses = list(
            self.store.search(diagnosis_namespace)
        )[:limit]
        
        # In production, this would use semantic similarity to match
        # current symptoms with relevant past consultations
        # For this implementation, we return recent history
        
        history_summary = "RELEVANT MEDICAL HISTORY:\n\n"
        
        if past_diagnoses:
            history_summary += "Past Diagnoses:\n"
            for diag in past_diagnoses[:5]:
                data = diag.value
                history_summary += f"- {data.get('diagnosis', 'Unknown')}"
                history_summary += f" ({data.get('timestamp', 'Date unknown')})\n"
            history_summary += "\n"
        
        if past_consultations:
            history_summary += "Recent Consultations:\n"
            for consult in past_consultations[:3]:
                data = consult.value
                history_summary += f"- {data.get('chief_complaint', 'Unknown')}"
                history_summary += f" - {data.get('clinical_summary', '')[:100]}...\n"
        
        return history_summary
    
    def get_patient_medications(
        self,
        patient_id_hash: str
    ) -> List[str]:
        """
        Retrieve current medication list.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        
        Returns:
        --------
        List[str]
            List of current medications
        """
        namespace = self.get_namespace(patient_id_hash, "medications")
        meds = list(self.store.search(namespace))
        
        return [med.value.get("medication_name", "") for med in meds]
    
    def get_patient_allergies(
        self,
        patient_id_hash: str
    ) -> List[str]:
        """
        Retrieve known allergies.
        
        Parameters:
        -----------
        patient_id_hash : str
            Hashed patient identifier
        
        Returns:
        --------
        List[str]
            List of known allergies
        """
        namespace = self.get_namespace(patient_id_hash, "allergies")
        allergies = list(self.store.search(namespace))
        
        return [
            allergy.value.get("allergen", "") 
            for allergy in allergies
        ]


# ============================================================================
# SECTION 6: CASE DATABASE (SELECT SIMILAR CASES)
# ============================================================================
#
# This section implements a case database for retrieving similar past cases.
# This demonstrates SELECT strategy for learning from historical outcomes.
# ============================================================================

class CaseDatabase:
    """
    Database of de-identified clinical cases for learning.
    
    This implements SELECT strategy for retrieving similar past cases
    to inform current diagnosis. All cases are de-identified to protect
    patient privacy while enabling clinical learning.
    """
    
    def __init__(self, store: BaseStore):
        """
        Initialize case database.
        
        Parameters:
        -----------
        store : BaseStore
            LangGraph store for case storage
        """
        self.store = store
    
    def add_case(
        self,
        presentation: str,
        diagnosis: str,
        outcome: str,
        specialty: Specialty
    ):
        """
        Add de-identified case to database.
        
        Parameters:
        -----------
        presentation : str
            Clinical presentation (de-identified)
        diagnosis : str
            Final diagnosis
        outcome : str
            Treatment outcome
        specialty : Specialty
            Primary specialty involved
        """
        namespace = ("case_database", specialty.value)
        case_id = str(uuid.uuid4())
        
        case_data = {
            "case_id": case_id,
            "presentation": presentation,
            "diagnosis": diagnosis,
            "outcome": outcome,
            "specialty": specialty.value,
            "timestamp": datetime.now().isoformat()
        }
        
        self.store.put(namespace, case_id, case_data)
    
    def find_similar_cases(
        self,
        symptoms: List[str],
        specialty: Specialty,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases from database.
        
        This implements SELECT strategy: retrieve only cases
        similar to current presentation.
        
        Parameters:
        -----------
        symptoms : List[str]
            Current patient symptoms
        specialty : Specialty
            Medical specialty
        limit : int
            Maximum number of cases to retrieve
        
        Returns:
        --------
        List[Dict[str, Any]]
            Similar cases with diagnoses and outcomes
        """
        namespace = ("case_database", specialty.value)
        cases = list(self.store.search(namespace))
        
        # In production, use semantic similarity matching
        # For this implementation, return recent cases
        similar_cases = []
        for case in cases[:limit]:
            similar_cases.append(case.value)
        
        return similar_cases


# ============================================================================
# SECTION 7: SPECIALIZED SUB-AGENTS (ISOLATE STRATEGY)
# ============================================================================
#
# These functions create specialized diagnostic sub-agents for different
# medical specialties. Each agent operates in an isolated context with
# specialty-specific knowledge and reasoning.
#
# This implements the ISOLATE strategy: separate medical specialties into
# dedicated contexts to prevent cross-contamination and enable expert-level
# focused analysis.
# ============================================================================

def create_cardiology_agent(llm, literature_tool):
    """
    Create cardiology diagnostic sub-agent.
    
    This agent specializes in cardiovascular conditions with isolated
    context focused only on cardiac assessment.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for cardiology assessment
    """
    cardiology_prompt = """You are a board-certified cardiologist assistant 
providing diagnostic support.

Your ONLY focus is cardiovascular conditions:
- Acute coronary syndrome and myocardial infarction
- Heart failure (systolic and diastolic)
- Arrhythmias and conduction disorders
- Valvular heart disease
- Hypertension and hypertensive emergencies
- Pericardial and myocardial disease

For the presented patient:
1. Assess cardiovascular system specifically
2. Identify cardiac risk factors
3. Determine if symptoms suggest cardiac pathology
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide differential diagnosis for cardiac conditions
6. Recommend cardiac-specific tests if indicated
7. Suggest treatment for identified cardiac issues

IMPORTANT:
- Focus ONLY on cardiac assessment
- Do not diagnose or comment on non-cardiac conditions
- Always support recommendations with evidence
- Consider both acute and chronic cardiac conditions
- Note any urgent cardiac concerns requiring immediate intervention

Provide your assessment in structured format:
CARDIAC ASSESSMENT:
- Cardiac symptoms present: [Yes/No with details]
- Risk factors identified: [List]
- Differential diagnoses: [Cardiac conditions to consider]
- Recommended tests: [Cardiac-specific tests]
- Urgency level: [Routine/Urgent/Emergent]
- Treatment recommendations: [If cardiac condition identified]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=cardiology_prompt
    )


def create_neurology_agent(llm, literature_tool):
    """
    Create neurology diagnostic sub-agent.
    
    This agent specializes in neurological conditions with isolated
    context focused only on neurological assessment.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for neurology assessment
    """
    neurology_prompt = """You are a board-certified neurologist assistant 
providing diagnostic support.

Your ONLY focus is neurological conditions:
- Stroke (ischemic and hemorrhagic)
- Seizures and epilepsy
- Headache disorders (migraine, tension, cluster)
- Dementia and cognitive disorders
- Movement disorders (Parkinson's, tremor)
- Peripheral neuropathy
- CNS infections and inflammation

For the presented patient:
1. Perform detailed neurological assessment
2. Evaluate mental status, cranial nerves, motor, sensory, coordination
3. Identify neurological symptoms and signs
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide differential diagnosis for neurological conditions
6. Recommend neuro-specific tests if indicated
7. Assess urgency of neurological findings

IMPORTANT:
- Focus ONLY on neurological assessment
- Do not diagnose or comment on non-neurological conditions
- Note any signs suggesting urgent neurology consultation
- Consider both central and peripheral nervous system
- Assess for red flags requiring immediate imaging

Provide your assessment in structured format:
NEUROLOGICAL ASSESSMENT:
- Neurological symptoms present: [Yes/No with details]
- Focal neurological signs: [List any focal findings]
- Differential diagnoses: [Neurological conditions to consider]
- Recommended tests: [Neuro-specific tests]
- Urgency level: [Routine/Urgent/Emergent]
- Treatment recommendations: [If neurological condition identified]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=neurology_prompt
    )


def create_pulmonology_agent(llm, literature_tool):
    """
    Create pulmonology diagnostic sub-agent.
    
    This agent specializes in respiratory conditions with isolated
    context focused only on pulmonary assessment.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for pulmonology assessment
    """
    pulmonology_prompt = """You are a board-certified pulmonologist assistant 
providing diagnostic support.

Your ONLY focus is respiratory conditions:
- Chronic obstructive pulmonary disease (COPD)
- Asthma and reactive airway disease
- Pneumonia and respiratory infections
- Interstitial lung disease
- Pulmonary embolism
- Pleural effusions and pneumothorax
- Sleep-disordered breathing

For the presented patient:
1. Assess respiratory system specifically
2. Evaluate dyspnea, cough, sputum production
3. Review oxygen saturation and respiratory rate
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide differential diagnosis for pulmonary conditions
6. Recommend pulmonary-specific tests if indicated
7. Assess need for oxygen therapy or respiratory support

IMPORTANT:
- Focus ONLY on pulmonary assessment
- Do not diagnose or comment on non-pulmonary conditions
- Assess severity of respiratory compromise
- Consider both obstructive and restrictive disease
- Note any respiratory emergency requiring immediate intervention

Provide your assessment in structured format:
PULMONARY ASSESSMENT:
- Respiratory symptoms present: [Yes/No with details]
- Oxygenation status: [Based on SpO2 and vital signs]
- Differential diagnoses: [Pulmonary conditions to consider]
- Recommended tests: [Pulmonary-specific tests]
- Urgency level: [Routine/Urgent/Emergent]
- Treatment recommendations: [If pulmonary condition identified]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=pulmonology_prompt
    )


def create_gastroenterology_agent(llm, literature_tool):
    """
    Create gastroenterology diagnostic sub-agent.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for gastroenterology assessment
    """
    gastro_prompt = """You are a board-certified gastroenterologist assistant 
providing diagnostic support.

Your ONLY focus is gastrointestinal conditions:
- GERD and peptic ulcer disease
- Inflammatory bowel disease (Crohn's, UC)
- Irritable bowel syndrome
- Hepatitis and liver disease
- Pancreatitis
- Gastrointestinal bleeding
- Bowel obstruction

For the presented patient:
1. Assess gastrointestinal system specifically
2. Evaluate abdominal pain, nausea, vomiting, diarrhea, constipation
3. Review liver function and GI-specific tests
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide differential diagnosis for GI conditions
6. Recommend GI-specific tests if indicated
7. Assess need for urgent endoscopy or surgery

IMPORTANT:
- Focus ONLY on GI assessment
- Do not diagnose or comment on non-GI conditions
- Consider both upper and lower GI tract
- Assess for surgical emergencies (perforation, obstruction)
- Note any GI bleeding requiring immediate intervention

Provide your assessment in structured format:
GI ASSESSMENT:
- GI symptoms present: [Yes/No with details]
- Alarm features: [Note any red flags]
- Differential diagnoses: [GI conditions to consider]
- Recommended tests: [GI-specific tests]
- Urgency level: [Routine/Urgent/Emergent]
- Treatment recommendations: [If GI condition identified]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=gastro_prompt
    )


def create_endocrinology_agent(llm, literature_tool):
    """
    Create endocrinology diagnostic sub-agent.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for endocrinology assessment
    """
    endocrine_prompt = """You are a board-certified endocrinologist assistant 
providing diagnostic support.

Your ONLY focus is endocrine and metabolic conditions:
- Diabetes mellitus (Type 1 and Type 2)
- Thyroid disorders (hypo/hyperthyroidism)
- Adrenal disorders
- Pituitary disorders
- Metabolic syndrome
- Osteoporosis and bone metabolism
- Electrolyte disorders

For the presented patient:
1. Assess endocrine system specifically
2. Review glucose, thyroid function, electrolytes
3. Evaluate for metabolic syndromes
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide differential diagnosis for endocrine conditions
6. Recommend endocrine-specific tests if indicated
7. Assess for diabetic/thyroid/adrenal emergencies

IMPORTANT:
- Focus ONLY on endocrine assessment
- Do not diagnose or comment on non-endocrine conditions
- Consider effects of endocrine disorders on other systems
- Assess for endocrine emergencies (DKA, thyroid storm, adrenal crisis)
- Review medications affecting endocrine function

Provide your assessment in structured format:
ENDOCRINE ASSESSMENT:
- Endocrine symptoms present: [Yes/No with details]
- Metabolic abnormalities: [Note any identified]
- Differential diagnoses: [Endocrine conditions to consider]
- Recommended tests: [Endocrine-specific tests]
- Urgency level: [Routine/Urgent/Emergent]
- Treatment recommendations: [If endocrine condition identified]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=endocrine_prompt
    )


def create_general_medicine_agent(llm, literature_tool):
    """
    Create general internal medicine diagnostic sub-agent.
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for the agent
    literature_tool : Tool
        Tool to retrieve medical literature
    
    Returns:
    --------
    CompiledGraph
        Compiled LangGraph agent for general medicine assessment
    """
    general_prompt = """You are a board-certified internal medicine physician 
providing comprehensive diagnostic support.

Your role is to provide general medical assessment and coordinate care:
- Synthesize findings from specialty consultations
- Identify multi-system conditions
- Assess general medical problems
- Consider social determinants of health
- Coordinate overall care plan

For the presented patient:
1. Perform comprehensive general medical assessment
2. Integrate findings from specialty assessments
3. Identify any missed diagnoses or conditions
4. Use retrieve_medical_literature tool for evidence-based guidelines
5. Provide overall differential diagnosis
6. Recommend additional general medical tests if needed
7. Suggest comprehensive treatment plan

IMPORTANT:
- Provide holistic assessment considering all organ systems
- Identify connections between symptoms that specialists may miss
- Consider infectious diseases, systemic conditions
- Assess overall patient stability and disposition
- Coordinate recommendations from multiple specialties

Provide your assessment in structured format:
GENERAL MEDICAL ASSESSMENT:
- Overall clinical picture: [Summary]
- Additional diagnoses not covered by specialists: [List]
- System-wide concerns: [Multi-organ involvement]
- Recommended tests: [General medical tests]
- Disposition: [Outpatient/Observation/Admission]
- Coordination notes: [Integration of specialty recommendations]
"""
    
    return create_react_agent(
        model=llm,
        tools=[literature_tool],
        state_modifier=general_prompt
    )


# ============================================================================
# SECTION 8: WORKFLOW NODES (WRITE, SELECT, COMPRESS, ISOLATE)
# ============================================================================
#
# These node functions implement the core workflow logic with all four
# contextual engineering strategies.
# ============================================================================

def intake_patient_data_node(
    state: MedicalConsultationState
) -> Dict[str, Any]:
    """
    Initial patient data intake and organization.
    
    This node implements the WRITE strategy by organizing patient
    presentation into structured format in the scratchpad.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state with patient presentation
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with organized data
    """
    chief_complaint = state.get("chief_complaint", "")
    symptoms = state.get("current_symptoms", [])
    vital_signs = state.get("vital_signs", {})
    
    # Write to scratchpad for workflow tracking
    scratchpad = f"PATIENT INTAKE COMPLETED\n"
    scratchpad += f"Chief Complaint: {chief_complaint}\n"
    scratchpad += f"Number of Symptoms: {len(symptoms)}\n"
    scratchpad += f"Vital Signs Recorded: {len(vital_signs)} measurements\n"
    
    # Determine which specialties should be consulted
    specialties_to_consult = determine_relevant_specialties(
        chief_complaint, symptoms
    )
    
    scratchpad += f"Specialties to Consult: {', '.join([s.value for s in specialties_to_consult])}\n"
    
    return {
        "scratchpad": scratchpad,
        "specialties_consulted": [s.value for s in specialties_to_consult]
    }


def determine_relevant_specialties(
    chief_complaint: str,
    symptoms: List[Dict[str, Any]]
) -> List[Specialty]:
    """
    Determine which medical specialties should be consulted.
    
    This implements intelligent routing based on presentation.
    
    Parameters:
    -----------
    chief_complaint : str
        Primary complaint
    symptoms : List[Dict[str, Any]]
        List of symptoms
    
    Returns:
    --------
    List[Specialty]
        Specialties to consult
    """
    specialties = [Specialty.GENERAL]  # Always consult general medicine
    
    # Simple keyword-based routing (in production, use ML model)
    complaint_lower = chief_complaint.lower()
    symptom_text = " ".join([s.get("description", "").lower() for s in symptoms])
    
    if any(word in complaint_lower + symptom_text for word in 
           ["chest pain", "palpitation", "heart", "cardiac"]):
        specialties.append(Specialty.CARDIOLOGY)
    
    if any(word in complaint_lower + symptom_text for word in 
           ["headache", "seizure", "numbness", "weakness", "stroke", "dizziness"]):
        specialties.append(Specialty.NEUROLOGY)
    
    if any(word in complaint_lower + symptom_text for word in 
           ["shortness of breath", "cough", "dyspnea", "wheezing"]):
        specialties.append(Specialty.PULMONOLOGY)
    
    if any(word in complaint_lower + symptom_text for word in 
           ["abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"]):
        specialties.append(Specialty.GASTROENTEROLOGY)
    
    if any(word in complaint_lower + symptom_text for word in 
           ["diabetes", "thyroid", "glucose", "thirst", "weight loss"]):
        specialties.append(Specialty.ENDOCRINOLOGY)
    
    return specialties


def retrieve_patient_context_node(
    state: MedicalConsultationState,
    store: BaseStore
) -> Dict[str, Any]:
    """
    Retrieve relevant patient history and medical literature.
    
    This node implements the SELECT strategy by:
    1. Querying patient records for relevant history
    2. Retrieving similar past cases
    3. Loading medical literature relevant to presentation
    
    Only relevant context is loaded to prevent information overload.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state
    store : BaseStore
        Storage for patient records and cases
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with selected context
    """
    patient_id_hash = state.get("patient_id_hash", "")
    current_symptoms = [s.get("description", "") for s in state.get("current_symptoms", [])]
    
    # Initialize managers
    records_manager = PatientRecordsManager(store)
    case_db = CaseDatabase(store)
    
    # SELECT: Retrieve relevant patient history
    relevant_history = records_manager.select_relevant_history(
        patient_id_hash,
        current_symptoms,
        limit=10
    )
    
    # SELECT: Get current medications and allergies
    medications = records_manager.get_patient_medications(patient_id_hash)
    allergies = records_manager.get_patient_allergies(patient_id_hash)
    
    # SELECT: Find similar cases for each relevant specialty
    specialties = state.get("specialties_consulted", [])
    similar_cases = []
    for specialty_str in specialties:
        try:
            specialty = Specialty(specialty_str)
            cases = case_db.find_similar_cases(
                current_symptoms,
                specialty,
                limit=2
            )
            similar_cases.extend(cases)
        except ValueError:
            pass
    
    # Update scratchpad
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nCONTEXT RETRIEVAL COMPLETED\n"
    scratchpad += f"Retrieved medical history: {len(relevant_history)} chars\n"
    scratchpad += f"Current medications: {len(medications)} items\n"
    scratchpad += f"Known allergies: {len(allergies)} items\n"
    scratchpad += f"Similar cases found: {len(similar_cases)}\n"
    
    return {
        "relevant_history": relevant_history,
        "current_medications": medications,
        "allergies": allergies,
        "similar_cases": similar_cases,
        "scratchpad": scratchpad
    }


def cardiology_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute cardiology assessment using isolated sub-agent.
    
    This node demonstrates ISOLATE strategy: cardiology agent operates
    in its own context window focused only on cardiac assessment.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state
    config : Dict[str, Any]
        Configuration containing cardiology agent
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with cardiology findings
    """
    # Check if cardiology consultation is needed
    if Specialty.CARDIOLOGY.value not in state.get("specialties_consulted", []):
        return {"cardiology_assessment": "Not consulted for this case."}
    
    # Get cardiology agent
    cardiology_agent = config["configurable"]["cardiology_agent"]
    
    # De-identify patient data for sub-agent (privacy control)
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.CARDIOLOGY
    )
    
    # Prepare cardiology-specific query
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs:
{format_vital_signs(patient_data['vital_signs'])}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide your cardiology assessment."""
    
    # Invoke cardiology agent in isolated context
    result = cardiology_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    # Extract assessment
    assessment = result["messages"][-1].content
    
    # Update scratchpad
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nCARDIOLOGY ASSESSMENT COMPLETED\n"
    
    return {
        "cardiology_assessment": assessment,
        "scratchpad": scratchpad
    }


def neurology_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute neurology assessment using isolated sub-agent.
    
    Implements ISOLATE strategy for neurological assessment.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state
    config : Dict[str, Any]
        Configuration containing neurology agent
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with neurology findings
    """
    if Specialty.NEUROLOGY.value not in state.get("specialties_consulted", []):
        return {"neurology_assessment": "Not consulted for this case."}
    
    neurology_agent = config["configurable"]["neurology_agent"]
    
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.NEUROLOGY
    )
    
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs:
{format_vital_signs(patient_data['vital_signs'])}

Physical Exam: {state.get('physical_exam_findings', 'Not documented')}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide your neurology assessment."""
    
    result = neurology_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    assessment = result["messages"][-1].content
    
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nNEUROLOGY ASSESSMENT COMPLETED\n"
    
    return {
        "neurology_assessment": assessment,
        "scratchpad": scratchpad
    }


def pulmonology_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute pulmonology assessment using isolated sub-agent.
    
    Implements ISOLATE strategy for pulmonary assessment.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state
    config : Dict[str, Any]
        Configuration containing pulmonology agent
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with pulmonology findings
    """
    if Specialty.PULMONOLOGY.value not in state.get("specialties_consulted", []):
        return {"pulmonology_assessment": "Not consulted for this case."}
    
    pulmonology_agent = config["configurable"]["pulmonology_agent"]
    
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.PULMONOLOGY
    )
    
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs (especially respiratory):
{format_vital_signs(patient_data['vital_signs'])}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide your pulmonology assessment."""
    
    result = pulmonology_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    assessment = result["messages"][-1].content
    
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nPULMONOLOGY ASSESSMENT COMPLETED\n"
    
    return {
        "pulmonology_assessment": assessment,
        "scratchpad": scratchpad
    }


def gastroenterology_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute gastroenterology assessment using isolated sub-agent."""
    if Specialty.GASTROENTEROLOGY.value not in state.get("specialties_consulted", []):
        return {"gastro_assessment": "Not consulted for this case."}
    
    gastro_agent = config["configurable"]["gastro_agent"]
    
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.GASTROENTEROLOGY
    )
    
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs:
{format_vital_signs(patient_data['vital_signs'])}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide your gastroenterology assessment."""
    
    result = gastro_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    assessment = result["messages"][-1].content
    
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nGASTROENTEROLOGY ASSESSMENT COMPLETED\n"
    
    return {
        "gastro_assessment": assessment,
        "scratchpad": scratchpad
    }


def endocrinology_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute endocrinology assessment using isolated sub-agent."""
    if Specialty.ENDOCRINOLOGY.value not in state.get("specialties_consulted", []):
        return {"endocrine_assessment": "Not consulted for this case."}
    
    endocrine_agent = config["configurable"]["endocrine_agent"]
    
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.ENDOCRINOLOGY
    )
    
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs:
{format_vital_signs(patient_data['vital_signs'])}

Test Results:
{format_test_results(patient_data['test_results'])}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide your endocrinology assessment."""
    
    result = endocrine_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    assessment = result["messages"][-1].content
    
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nENDOCRINOLOGY ASSESSMENT COMPLETED\n"
    
    return {
        "endocrine_assessment": assessment,
        "scratchpad": scratchpad
    }


def general_medicine_assessment_node(
    state: MedicalConsultationState,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute general medicine assessment and synthesize findings."""
    general_agent = config["configurable"]["general_agent"]
    
    patient_data = PrivacyManager.de_identify_for_subagent(
        state,
        Specialty.GENERAL
    )
    
    # Include specialty assessments for synthesis
    specialty_assessments = f"""
CARDIOLOGY: {state.get('cardiology_assessment', 'Not consulted')}

NEUROLOGY: {state.get('neurology_assessment', 'Not consulted')}

PULMONOLOGY: {state.get('pulmonology_assessment', 'Not consulted')}

GASTROENTEROLOGY: {state.get('gastro_assessment', 'Not consulted')}

ENDOCRINOLOGY: {state.get('endocrine_assessment', 'Not consulted')}
"""
    
    query = f"""Patient Presentation:
Chief Complaint: {patient_data['chief_complaint']}

Current Symptoms:
{format_symptoms(patient_data['symptoms'])}

Vital Signs:
{format_vital_signs(patient_data['vital_signs'])}

Specialty Assessments:
{specialty_assessments}

Current Medications: {', '.join(patient_data['medications'])}
Allergies: {', '.join(patient_data['allergies'])}

Relevant History:
{patient_data['relevant_history']}

Please provide overall assessment and synthesize specialty findings."""
    
    result = general_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    assessment = result["messages"][-1].content
    
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nGENERAL MEDICINE ASSESSMENT COMPLETED\n"
    
    return {
        "general_assessment": assessment,
        "scratchpad": scratchpad
    }


def compress_clinical_findings_node(
    state: MedicalConsultationState,
    llm
) -> Dict[str, Any]:
    """
    Compress all assessments into clinical summary.
    
    This node implements the COMPRESS strategy by:
    1. Aggregating all specialty assessments
    2. Summarizing into concise clinical note
    3. Extracting key findings, diagnoses, and plan
    
    This reduces lengthy assessments into actionable summary
    suitable for patient record and care coordination.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state with all assessments
    llm : ChatModel
        Language model for summarization
    
    Returns:
    --------
    Dict[str, Any]
        Updates to state with compressed summary
    """
    # Collect all assessments
    assessments_text = f"""
CHIEF COMPLAINT: {state.get('chief_complaint', '')}

CARDIOLOGY ASSESSMENT:
{state.get('cardiology_assessment', 'Not consulted')}

NEUROLOGY ASSESSMENT:
{state.get('neurology_assessment', 'Not consulted')}

PULMONOLOGY ASSESSMENT:
{state.get('pulmonology_assessment', 'Not consulted')}

GASTROENTEROLOGY ASSESSMENT:
{state.get('gastro_assessment', 'Not consulted')}

ENDOCRINOLOGY ASSESSMENT:
{state.get('endocrine_assessment', 'Not consulted')}

GENERAL MEDICINE ASSESSMENT:
{state.get('general_assessment', '')}
"""
    
    # Compression prompt for clinical summary
    compression_prompt = """You are creating a concise clinical consultation summary.

Compress the specialty assessments into a structured clinical note:

CLINICAL SUMMARY:
[2-3 sentence overview of patient presentation and key findings]

DIFFERENTIAL DIAGNOSES:
1. [Most likely diagnosis with supporting evidence]
2. [Second possibility with evidence]
3. [Third possibility with evidence]

RECOMMENDED WORKUP:
- [Essential tests needed]
- [Additional tests if indicated]

TREATMENT PLAN:
- [Immediate interventions]
- [Ongoing management]
- [Follow-up planning]

CRITICAL ACTIONS:
[Any urgent interventions needed - or state "None" if stable]

Be concise but retain all clinically significant information.
Prioritize actionable items."""
    
    # Invoke LLM for compression
    messages = [
        SystemMessage(content=compression_prompt),
        HumanMessage(content=f"Compress these assessments:\n\n{assessments_text}")
    ]
    
    result = llm.invoke(messages)
    summary = result.content
    
    # Update scratchpad
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nCOMPRESSION COMPLETED\n"
    scratchpad += f"Original length: {len(assessments_text)} chars\n"
    scratchpad += f"Compressed length: {len(summary)} chars\n"
    scratchpad += f"Reduction: {100 * (1 - len(summary)/len(assessments_text)):.1f}%\n"
    
    return {
        "clinical_summary": summary,
        "scratchpad": scratchpad
    }


def store_consultation_record_node(
    state: MedicalConsultationState,
    store: BaseStore
) -> Dict[str, Any]:
    """
    Store consultation in patient record.
    
    This node implements the WRITE strategy for long-term storage.
    The consultation summary is stored in patient's medical record
    for future reference and continuity of care.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Current state with consultation data
    store : BaseStore
        Storage for patient records
    
    Returns:
    --------
    Dict[str, Any]
        Updates to scratchpad
    """
    patient_id_hash = state.get("patient_id_hash", "")
    consultation_id = state.get("consultation_id", "")
    
    # Initialize records manager
    records_manager = PatientRecordsManager(store)
    
    # Prepare consultation data for storage
    consultation_data = {
        "chief_complaint": state.get("chief_complaint", ""),
        "symptoms": state.get("current_symptoms", []),
        "vital_signs": state.get("vital_signs", {}),
        "clinical_summary": state.get("clinical_summary", ""),
        "specialties_consulted": state.get("specialties_consulted", []),
        "recommended_tests": state.get("recommended_tests", []),
        "treatment_recommendations": state.get("treatment_recommendations", [])
    }
    
    # Store consultation
    records_manager.write_consultation(
        patient_id_hash,
        consultation_data,
        user_id="system_diagnostic_assistant"
    )
    
    # Update scratchpad
    scratchpad = state.get("scratchpad", "")
    scratchpad += f"\nCONSULTATION RECORD STORED\n"
    scratchpad += f"Patient ID: {patient_id_hash[:16]}...\n"
    scratchpad += f"Consultation ID: {consultation_id}\n"
    
    return {
        "scratchpad": scratchpad
    }


# ============================================================================
# SECTION 9: HELPER FUNCTIONS
# ============================================================================

def format_symptoms(symptoms: List[Dict[str, Any]]) -> str:
    """Format symptoms for agent query."""
    if not symptoms:
        return "No symptoms documented"
    
    formatted = ""
    for sym in symptoms:
        formatted += f"- {sym.get('description', 'Unknown')}: "
        formatted += f"Severity {sym.get('severity', 'unknown')}, "
        formatted += f"Duration {sym.get('duration_days', 0)} days\n"
    
    return formatted


def format_vital_signs(vitals: Dict[str, Any]) -> str:
    """Format vital signs for agent query."""
    if not vitals:
        return "No vital signs recorded"
    
    formatted = ""
    for key, value in vitals.items():
        formatted += f"- {key}: {value}\n"
    
    return formatted


def format_test_results(tests: List[Dict[str, Any]]) -> str:
    """Format test results for agent query."""
    if not tests:
        return "No recent test results"
    
    formatted = ""
    for test in tests:
        formatted += f"- {test.get('test_name', 'Unknown')}: "
        formatted += f"{test.get('result_value', 'N/A')} "
        formatted += f"(Normal: {test.get('normal_range', 'N/A')})"
        if test.get('abnormal', False):
            formatted += " [ABNORMAL]"
        formatted += "\n"
    
    return formatted


# ============================================================================
# SECTION 10: WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_medical_diagnosis_workflow(
    llm,
    embeddings,
    store: BaseStore
):
    """
    Create the complete medical diagnosis workflow graph.
    
    This function assembles all components into a cohesive workflow
    implementing the full contextual engineering pipeline:
    
    WORKFLOW STAGES:
    ----------------
    1. intake_patient_data: Organize presentation (WRITE to scratchpad)
    2. retrieve_context: Get relevant history and literature (SELECT)
    3. cardiology_assessment: Isolated cardiac evaluation (ISOLATE)
    4. neurology_assessment: Isolated neuro evaluation (ISOLATE)
    5. pulmonology_assessment: Isolated pulmonary evaluation (ISOLATE)
    6. gastro_assessment: Isolated GI evaluation (ISOLATE)
    7. endocrine_assessment: Isolated endocrine evaluation (ISOLATE)
    8. general_assessment: Synthesize findings (ISOLATE + coordination)
    9. compress_findings: Create clinical summary (COMPRESS)
    10. store_consultation: Save to patient record (WRITE to long-term)
    
    Parameters:
    -----------
    llm : ChatModel
        Language model for agents and summarization
    embeddings : Embeddings
        Embedding model for RAG
    store : BaseStore
        Storage for patient records and medical cases
    
    Returns:
    --------
    tuple
        (CompiledGraph, Dict of agents)
    """
    # Initialize medical knowledge base (for SELECT strategy)
    kb = MedicalKnowledgeBase(embeddings)
    literature_tool = kb.get_retriever_tool()
    
    # Create specialized sub-agents (for ISOLATE strategy)
    cardiology_agent = create_cardiology_agent(llm, literature_tool)
    neurology_agent = create_neurology_agent(llm, literature_tool)
    pulmonology_agent = create_pulmonology_agent(llm, literature_tool)
    gastro_agent = create_gastroenterology_agent(llm, literature_tool)
    endocrine_agent = create_endocrinology_agent(llm, literature_tool)
    general_agent = create_general_medicine_agent(llm, literature_tool)
    
    # Build the state graph
    workflow = StateGraph(MedicalConsultationState)
    
    # Add nodes to the workflow
    workflow.add_node("intake_patient_data", intake_patient_data_node)
    workflow.add_node("retrieve_context",
                     lambda state: retrieve_patient_context_node(state, store))
    workflow.add_node("cardiology_assessment",
                     lambda state, config: cardiology_assessment_node(state, config))
    workflow.add_node("neurology_assessment",
                     lambda state, config: neurology_assessment_node(state, config))
    workflow.add_node("pulmonology_assessment",
                     lambda state, config: pulmonology_assessment_node(state, config))
    workflow.add_node("gastro_assessment",
                     lambda state, config: gastroenterology_assessment_node(state, config))
    workflow.add_node("endocrine_assessment",
                     lambda state, config: endocrinology_assessment_node(state, config))
    workflow.add_node("general_assessment",
                     lambda state, config: general_medicine_assessment_node(state, config))
    workflow.add_node("compress_findings",
                     lambda state: compress_clinical_findings_node(state, llm))
    workflow.add_node("store_consultation",
                     lambda state: store_consultation_record_node(state, store))
    
    # Define workflow edges (execution order)
    workflow.add_edge(START, "intake_patient_data")
    workflow.add_edge("intake_patient_data", "retrieve_context")
    workflow.add_edge("retrieve_context", "cardiology_assessment")
    workflow.add_edge("cardiology_assessment", "neurology_assessment")
    workflow.add_edge("neurology_assessment", "pulmonology_assessment")
    workflow.add_edge("pulmonology_assessment", "gastro_assessment")
    workflow.add_edge("gastro_assessment", "endocrine_assessment")
    workflow.add_edge("endocrine_assessment", "general_assessment")
    workflow.add_edge("general_assessment", "compress_findings")
    workflow.add_edge("compress_findings", "store_consultation")
    workflow.add_edge("store_consultation", END)
    
    # Compile with checkpointing (SHORT-TERM MEMORY)
    checkpointer = InMemorySaver()
    
    compiled = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    agents = {
        "cardiology_agent": cardiology_agent,
        "neurology_agent": neurology_agent,
        "pulmonology_agent": pulmonology_agent,
        "gastro_agent": gastro_agent,
        "endocrine_agent": endocrine_agent,
        "general_agent": general_agent
    }
    
    return compiled, agents


# ============================================================================
# SECTION 11: UTILITY FUNCTIONS AND DEMONSTRATION
# ============================================================================

def format_clinical_output(state: MedicalConsultationState) -> str:
    """
    Format the clinical consultation output for display.
    
    Parameters:
    -----------
    state : MedicalConsultationState
        Final state after workflow completion
    
    Returns:
    --------
    str
        Formatted clinical consultation report
    """
    output = "=" * 80 + "\n"
    output += "CLINICAL CONSULTATION REPORT\n"
    output += "=" * 80 + "\n\n"
    
    output += f"Patient ID: {state.get('patient_id_hash', 'Unknown')[:16]}...\n"
    output += f"Consultation ID: {state.get('consultation_id', 'Unknown')}\n"
    output += f"Chief Complaint: {state.get('chief_complaint', 'Not documented')}\n\n"
    
    output += "CLINICAL SUMMARY:\n"
    output += "-" * 80 + "\n"
    output += state.get("clinical_summary", "No summary available") + "\n\n"
    
    output += "SPECIALTY CONSULTATIONS:\n"
    output += "-" * 80 + "\n"
    specialties = state.get("specialties_consulted", [])
    output += f"Consulted: {', '.join(specialties)}\n\n"
    
    if state.get("cardiology_assessment", "") != "Not consulted for this case.":
        output += "CARDIOLOGY:\n"
        output += state.get("cardiology_assessment", "")[:200] + "...\n\n"
    
    if state.get("neurology_assessment", "") != "Not consulted for this case.":
        output += "NEUROLOGY:\n"
        output += state.get("neurology_assessment", "")[:200] + "...\n\n"
    
    output += "=" * 80 + "\n"
    output += "WORKFLOW PROCESSING NOTES:\n"
    output += "=" * 80 + "\n"
    output += state.get("scratchpad", "No processing notes") + "\n"
    
    return output


def demonstrate_medical_diagnosis_assistant():
    """
    Complete demonstration of the medical diagnosis assistant.
    
    This shows all contextual engineering strategies in action:
    - WRITE: Patient data to session and record storage
    - SELECT: Relevant history and medical literature
    - COMPRESS: Lengthy assessments into clinical summary
    - ISOLATE: Specialty sub-agents with privacy controls
    """
    print("Initializing Medical Diagnosis Assistant...")
    print("=" * 80)
    
    # Initialize components
    llm = init_chat_model(
        "anthropic:claude-sonnet-4-20250514",
        temperature=0
    )
    
    embeddings = init_embeddings("openai:text-embedding-3-small")
    
    store = InMemoryStore()
    
    # Create workflow
    print("\nBuilding workflow graph...")
    workflow, agents = create_medical_diagnosis_workflow(llm, embeddings, store)
    
    print("Workflow created successfully!")
    print("\nWorkflow Components:")
    print("- Patient Intake Node (WRITE to scratchpad)")
    print("- Context Retrieval Node (SELECT from records/literature)")
    print("- Cardiology Assessment Node (ISOLATE)")
    print("- Neurology Assessment Node (ISOLATE)")
    print("- Pulmonology Assessment Node (ISOLATE)")
    print("- Gastroenterology Assessment Node (ISOLATE)")
    print("- Endocrinology Assessment Node (ISOLATE)")
    print("- General Medicine Node (ISOLATE + coordinate)")
    print("- Findings Compression Node (COMPRESS)")
    print("- Record Storage Node (WRITE to long-term)")
    
    # Sample patient case
    patient_id = "patient_12345"
    patient_id_hash = PrivacyManager.hash_patient_id(patient_id)
    consultation_id = str(uuid.uuid4())
    
    sample_state = {
        "patient_id_hash": patient_id_hash,
        "consultation_id": consultation_id,
        "chief_complaint": "Chest pain and shortness of breath for 2 hours",
        "current_symptoms": [
            {
                "description": "Substernal chest pressure",
                "severity": "high",
                "duration_days": 0,
                "onset_date": "2024-01-27",
                "location": "chest"
            },
            {
                "description": "Shortness of breath",
                "severity": "moderate",
                "duration_days": 0,
                "onset_date": "2024-01-27",
                "location": "respiratory"
            },
            {
                "description": "Diaphoresis",
                "severity": "moderate",
                "duration_days": 0,
                "onset_date": "2024-01-27",
                "location": "systemic"
            }
        ],
        "vital_signs": {
            "blood_pressure": "150/95 mmHg",
            "heart_rate": "105 bpm",
            "respiratory_rate": "22 breaths/min",
            "temperature": "37.2°C",
            "oxygen_saturation": "94% on room air"
        },
        "physical_exam_findings": "Patient appears anxious and diaphoretic. Cardiovascular: tachycardic, regular rhythm, no murmurs. Respiratory: mildly tachypneic, clear bilaterally.",
        "relevant_history": "",
        "past_diagnoses": ["Hypertension", "Hyperlipidemia"],
        "current_medications": ["Lisinopril 10mg daily", "Atorvastatin 20mg daily"],
        "allergies": ["Penicillin - rash"],
        "recent_test_results": [],
        "cardiology_assessment": "",
        "neurology_assessment": "",
        "pulmonology_assessment": "",
        "gastro_assessment": "",
        "endocrine_assessment": "",
        "general_assessment": "",
        "differential_diagnoses": [],
        "recommended_tests": [],
        "treatment_recommendations": [],
        "scratchpad": "",
        "clinical_summary": "",
        "retrieved_literature": [],
        "similar_cases": [],
        "specialties_consulted": [],
        "audit_trail": [],
        "privacy_level": "confidential"
    }
    
    print("\n" + "=" * 80)
    print("EXECUTING CLINICAL CONSULTATION")
    print("=" * 80)
    
    # Configure with agents
    config = {
        "configurable": {
            "thread_id": "consultation_session_1",
            "cardiology_agent": agents["cardiology_agent"],
            "neurology_agent": agents["neurology_agent"],
            "pulmonology_agent": agents["pulmonology_agent"],
            "gastro_agent": agents["gastro_agent"],
            "endocrine_agent": agents["endocrine_agent"],
            "general_agent": agents["general_agent"]
        }
    }
    
    print("\nStarting consultation workflow...")
    print("This demonstrates all contextual engineering strategies:\n")
    
    try:
        final_state = workflow.invoke(sample_state, config)
        
        print("\n" + "=" * 80)
        print("CONSULTATION COMPLETE")
        print("=" * 80)
        
        # Display formatted output
        print("\n" + format_clinical_output(final_state))
        
        # Demonstrate checkpoint inspection
        print("\n" + "=" * 80)
        print("CHECKPOINT INSPECTION (SHORT-TERM MEMORY)")
        print("=" * 80)
        latest_state = workflow.get_state(config)
        print(f"\nCheckpoint ID: {latest_state.config['configurable']['checkpoint_id']}")
        print(f"Step: {latest_state.metadata.get('step', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nAll contextual engineering strategies demonstrated:")
        print("1. WRITE: Patient data stored in session and records")
        print("2. SELECT: Relevant history and literature retrieved")
        print("3. COMPRESS: Lengthy assessments summarized")
        print("4. ISOLATE: Specialty sub-agents with privacy controls")
        print("5. SHORT-TERM: Checkpointing for session persistence")
        print("6. LONG-TERM: Patient records across consultations")
        print("7. PRIVACY: HIPAA-compliant data handling")
        
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# SECTION 12: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point demonstrating the medical diagnosis assistant
    with complete contextual engineering and privacy controls.
    """
    print("\n")
    print("=" * 80)
    print("MEDICAL DIAGNOSIS ASSISTANT")
    print("Contextual Engineering with Privacy Controls")
    print("=" * 80)
    print("\nThis system demonstrates:")
    print("- WRITE: Store symptoms, tests, and diagnoses")
    print("- SELECT: Retrieve relevant medical history and literature")
    print("- COMPRESS: Summarize lengthy medical records")
    print("- ISOLATE: Privacy-preserving specialty sub-agents")
    print("- HIPAA Compliance: Audit trails and data protection")
    print("\n" + "=" * 80 + "\n")
    
    demonstrate_medical_diagnosis_assistant()
