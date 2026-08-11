\section{Methodology}
\label{sec:methodology}

This section presents the research methodology, architectural design rationale, cognitive decomposition, and implementation strategy underlying the Agricultural Cognitive Architecture (ACA). ACA is a layered, cognition-oriented software architecture for autonomous precision farming that decouples cognitive reasoning, memory management, knowledge retrieval, world modelling, skill composition, and tool interaction from physical deployment environments. The presentation proceeds from the research questions that motivated the architecture, through the design process that shaped it, to the verification strategy that validates its structural and behavioural properties. All claims are grounded in the realised reference implementation; components that exist only as architectural extension points or interface-level scaffolds are explicitly identified as such.

% ============================================================
\subsection{Research Methodology}
\label{sec:research_methodology}

The research methodology follows a design-science paradigm \cite{hevner2004design}, wherein the primary contribution is a purposefully designed artefact---a reusable cognitive architecture---and the evaluation criteria are structural soundness, contract enforcement, and extensibility rather than field-level agricultural efficacy, which is deferred to future domain-specific evaluation campaigns.

\subsubsection{Research Problem}
Precision agriculture demands sustained, multi-timescale reasoning: real-time sensor fusion at sub-second granularity, medium-term planning under environmental uncertainty, and long-term learning from intervention outcomes spanning multiple growing seasons. Monolithic controllers conflate these temporally and functionally distinct concerns into tightly coupled systems that resist extension, hinder auditability, and preclude formal explainability. The central research question is therefore:

\begin{quote}
\emph{Can a reusable cognitive architecture, modelled after established theories of cognitive systems \cite{laird2012soar, anderson2004actr}, provide a principled software foundation for autonomous agricultural decision-making that is simultaneously domain-agnostic in its cognitive infrastructure and domain-specific only at its extension points?}
\end{quote}

\subsubsection{Design Objectives}
Three design objectives guide the architecture. First, the architecture must enforce a \emph{separation of cognitive concerns}, isolating perception, reasoning, planning, execution, learning, and meta-cognition into independently evolvable subsystems. Second, the architecture must support \emph{end-to-end explainability}: every decision must be traceable through a formal provenance chain linking observations, evidence, hypotheses, beliefs, and actions. Third, the architecture must be \emph{deployment-agnostic}, enabling the same cognitive pipeline to execute on resource-constrained edge devices or cloud infrastructure without modification to any cognitive component.

\subsubsection{Research Phases}
The research proceeds in five phases. Phase~I establishes requirements through an analysis of the agricultural decision-making domain and the limitations of existing architectures. Phase~II defines a layered cognitive decomposition grounded in cognitive science, specifying the interfaces and data flows among subsystems. Phase~III realises the decomposition as a dependency-injected, message-driven software architecture with formal agent contracts. Phase~IV implements the reference architecture in Python, deliberately restricting external dependencies to the Python standard library and NumPy to ensure broad portability. Phase~V validates the cognitive infrastructure through a comprehensive unit test suite spanning 223 test methods across five milestone-organised test modules.

A strict separation is maintained between \emph{cognitive infrastructure validation} and \emph{domain-specific model validation}. Agricultural AI models (e.g., crop disease classifiers, yield predictors, pest detection networks) are intentionally excluded from the current scope. The architecture validates the reusable cognitive substrate independently from domain intelligence, which is treated as pluggable and injected through well-defined tool, skill, and knowledge interfaces.

% ============================================================
\subsection{System Design Process}
\label{sec:system_design}

The system design process follows a contract-first, interface-driven methodology in which abstract specifications are defined before any concrete implementation is written. 

\subsubsection{Architectural Principles}
Table~\ref{tab:design_principles} summarises the governing principles and their concrete realisation in the architecture.

\begin{table}[htbp]
\centering
\caption{Architectural principles governing ACA and their realisation in the reference implementation.}
\label{tab:design_principles}
% Wrapped in resizebox to prevent Overfull/Underfull hboxes in IEEE's narrow columns
\resizebox{\columnwidth}{!}{%
\begin{tabular}{p{4.0cm} p{9.0cm}}
\toprule
\textbf{Principle} & \textbf{Realisation} \\
\midrule
Layered architecture & Six cognitive layers with unidirectional primary data flow and feedback loops. \\
Loose coupling & All inter-component communication through a typed pub/sub message bus. \\
Dependency inversion & Components depend on abstract interfaces (\texttt{ABCMeta}); concrete implementations injected at construction. \\
Interface segregation & Separate abstract base classes for tools, skills, agents, world model, digital twin, embedders, and vector stores. \\
Immutable configuration & All configuration objects are frozen dataclasses; runtime mutation is structurally prevented. \\
Contract enforcement & Agents declare formal contracts specifying permitted memory modules, tools, and message types. \\
Component isolation & Memory subsystems, knowledge stores, and cognitive layers are independently instantiable and testable. \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsubsection{Design Patterns and SOLID Principles}
The architecture employs several standard design patterns to ensure robustness. The \emph{Abstract Factory} pattern governs object creation via \texttt{ACAConfig.load()}. The \emph{Strategy} pattern appears in the Reasoning layer's injectable hypothesis generator and the Scheduler's pluggable scheduling policy. The \emph{Registry} pattern centralises tool and skill discovery, while the \emph{Proxy} pattern (\texttt{MemoryGateway} and \texttt{ToolGateway}) wraps real subsystems with permission-checked facades. The \emph{Observer} pattern underpins all inter-component communication through the \texttt{MessageBus}. 

The architecture demonstrates strict adherence to the SOLID principles. \emph{Single Responsibility} ensures each module addresses exactly one concern. \emph{Open/Closed} allows new capabilities to be added via registries without modifying existing components. \emph{Liskov Substitution} ensures all abstract hierarchies enforce method signatures, making conforming subclasses perfectly substitutable.

% ============================================================
\begin{figure*}[htbp]
  \centering
  \resizebox{\textwidth}{!}{%
  \begin{tikzpicture}[
      font=\sffamily\small,
      >=Latex,
      % Color Palette Definition
      envcol/.style={fill=gray!15, draw=gray!80},
      perccol/.style={fill=green!10, draw=green!60!black},
      corecol/.style={fill=blue!10, draw=blue!60!black},
      subcol/.style={fill=violet!10, draw=violet!60!black},
      actcol/.style={fill=orange!15, draw=orange!70!black},
      % Node Styles
      layerbox/.style={draw, rounded corners=3pt, align=center, font=\sffamily\bfseries\footnotesize},
      module/.style={draw, rounded corners=2pt, align=center, inner sep=4pt, minimum height=1.1cm, text width=2.5cm, font=\sffamily\footnotesize},
      bus/.style={draw=gray!60, dashed, fill=gray!5, rounded corners=2pt, minimum height=0.8cm, minimum width=15cm, align=center, font=\sffamily\scriptsize\itshape},
      % Arrow Styles
      dataflow/.style={->, thick, draw=black!80},
      feedback/.style={->, dashed, thick, draw=black!60},
      comm/.style={-, thin, draw=gray!80}
    ]

    % ==========================================
    % LAYER 1: ENVIRONMENT
    % ==========================================
    \node[layerbox, envcol, minimum width=15.4cm, inner sep=8pt] (env) at (0, 0) {LAYER 1 -- AGRICULTURAL ENVIRONMENT\\ \normalfont\scriptsize Crops, Soil, Weather, Farm Conditions, Sensors, IoT \& UAV Observations};

    % ==========================================
    % LAYER 2: PERCEPTION
    % ==========================================
    \node[layerbox, perccol, minimum width=12cm, inner sep=8pt] (layer2) at (0, -1.8) {LAYER 2 -- PERCEPTION / STATE ESTIMATION};
    
    % ==========================================
    % LAYER 3: COGNITIVE CORE (Expanded Grid)
    % ==========================================
    \node[module, subcol] (meta) at (0, -3.3) {Meta-Cognition};
    
    % Widened the X-coordinates to give text room to breathe
    \node[module, corecol] (perc) at (-6.6, -5.2) {Perception};
    \node[module, corecol] (reason) at (-2.2, -5.2) {Reasoning};
    \node[module, corecol] (plan) at (2.2, -5.2) {Planning};
    \node[module, corecol] (learn) at (6.6, -5.2) {Learning};

    % ==========================================
    % LAYER 4: COGNITIVE SUBSTRATES
    % ==========================================
    \node[module, subcol] (memory) at (-6.6, -8.4) {Memory Subsystems\\\scriptsize(Working/Episodic)};
    \node[module, subcol] (kb) at (-2.2, -8.4) {External Knowledge\\\scriptsize(Agentic RAG / KB)};
    \node[module, subcol] (world) at (2.2, -8.4) {World Model\\\scriptsize(State Rep.)};
    \node[module, subcol] (twin) at (6.6, -8.4) {Digital Twin\\\scriptsize(Simulation/Predictive)};

    % ==========================================
    % MESSAGE BUS & LAYER 5
    % ==========================================
    \node[bus] (bus) at (0, -10.6) {Message-Driven Orchestration / Communication Substrate\\(Typed Contracts, Publish/Subscribe)};
    
    \node[layerbox, actcol, minimum width=12cm, inner sep=8pt] (action) at (0, -12.4) {LAYER 5 -- ACTION / EXECUTION\\ \normalfont\scriptsize Tool Layer $\rightarrow$ Skill Layer $\rightarrow$ Agricultural Actuators};

    % ==========================================
    % BACKGROUND CONTAINERS (Strictly Bounded)
    % ==========================================
    \begin{scope}[on background layer]
      % Core Background
      \coordinate (core_tl) at (-8.3, -2.5);
      \coordinate (core_br) at (8.3, -6.4);
      \node[layerbox, fill=blue!5, draw=blue!30, fit=(core_tl) (core_br)] (corebox) {};
      \node[anchor=north west, inner sep=6pt, font=\sffamily\bfseries\footnotesize] at (corebox.north west) {LAYER 3 -- COGNITIVE CORE};

      % Substrate Background
      \coordinate (sub_tl) at (-8.3, -6.8);
      \coordinate (sub_br) at (8.3, -9.8);
      \node[layerbox, fill=violet!5, draw=violet!30, fit=(sub_tl) (sub_br)] (subbox) {};
      \node[anchor=north west, inner sep=6pt, font=\sffamily\bfseries\footnotesize] at (subbox.north west) {LAYER 4 -- COGNITIVE SUBSTRATES};

      % ACA Master Background 
      \coordinate (aca_tl) at (-8.8, -0.8);
      \coordinate (aca_br) at (8.8, -11.4);
      \node[draw=gray!50, dashed, thick, rounded corners=5pt, fit=(aca_tl) (aca_br)] (acabox) {};
      
      % Multi-line title anchored top-left to avoid center arrow
      \node[anchor=north west, font=\sffamily\bfseries, text=gray!70, align=left] at ([shift={(8pt,-8pt)}]acabox.north west) {AGRICULTURAL COGNITIVE\\[-0.5ex]ARCHITECTURE (ACA)};
    \end{scope}

    % ==========================================
    % ROUTING & ARROWS
    % ==========================================
    
    % Global Data Flow (White fill masks the dashed line perfectly)
    \draw[dataflow] (env.south) -- node[right=2pt, fill=white, inner sep=2pt, font=\scriptsize] {Raw Observations} (layer2.north);
    
    % State Rep sits safely on the outside (left) of the arrow
    \draw[dataflow] (layer2.south -| perc.north) -- node[left=3pt, font=\scriptsize] {State Rep.} (perc.north);
    
    \draw[dataflow] (bus.south) -- node[right, font=\scriptsize] {Executable Actions} (action.north);

    % Core Internal Flow (Now with enough horizontal space and pushed upwards)
    \draw[dataflow] (perc.east) -- node[above=2pt, font=\scriptsize, align=center] {Structured\\Evidence} (reason.west);
    \draw[dataflow] (reason.east) -- node[above=2pt, font=\scriptsize, align=center] {Probabilistic\\Beliefs} (plan.west);
    \draw[dataflow] (plan.east) -- node[above=2pt, font=\scriptsize, align=center] {Justified\\Decisions} (learn.west);
    
    % Plan bypassing to the Bus (Threads the needle precisely through the X=0 gap)
    \draw[dataflow, rounded corners=4pt] ([xshift=-15pt]plan.south) -- ++(0,-0.7) -| (0, -10.6 |- bus.north);

    % Meta-cognition supervision (Explicit point-to-point routing avoids text)
    \draw[feedback] (meta.south west) -- (perc.north);
    \draw[feedback] ([xshift=-15pt]meta.south) -- (reason.north);
    \draw[feedback] ([xshift=15pt]meta.south) -- (plan.north);
    \draw[feedback] (meta.south east) -- (learn.north);

    % Substrate Interactions (Straight vertical drops)
    \draw[comm] (perc.south) -- (memory.north);
    \draw[comm] (reason.south) -- (kb.north);
    \draw[comm] ([xshift=15pt]plan.south) -- ([xshift=15pt]world.north); % Safely shifted right
    \draw[comm] (learn.south) -- (twin.north);

    % Grand Feedback Loops (Pushed out wide to prevent layer label overlapping)
    \draw[feedback, rounded corners=5pt] (action.west) -- (-10.4, -12.4) |- node[pos=0.25, left=3pt, align=right, font=\scriptsize] {Observed Outcomes\\(Continuous Feedback)} (layer2.west);
    
    \draw[feedback, rounded corners=5pt] (action.east) -- (10.4, -12.4) |- node[pos=0.25, right=3pt, align=left, font=\scriptsize] {Learning \&\\Adaptation} (learn.east);

  \end{tikzpicture}%
  }
  \caption{Overall architecture of the Agricultural Cognitive Architecture (ACA). The system operates as a closed cognitive loop rather than a linear pipeline. Raw environmental observations (Layer 1) are transformed into structured state representations (Layer 2), which are continually reasoned upon by the Cognitive Core (Layer 3) supported by persistent memory and knowledge substrates (Layer 4). Meta-cognition supervises the cognitive loop, while a message-driven substrate orchestrates execution through the tool-skill hierarchy (Layer 5). Observed outcomes form a continuous feedback loop that drives learning and adaptation (Layer 6).}
  \label{fig:aca_architecture}
\end{figure*}
% ============================================================
\subsection{Cognitive Layer Design}
\label{sec:cognitive_layers}

% ---- Perception ----
\subsubsection{Perception Layer}
\label{sec:perception}
The Perception layer transforms raw sensor telemetry into validated, normalised feature objects. The design decomposes perception into three collaborating components: an \emph{Observation Validator}, an \emph{Observation Normaliser}, and an \emph{Observation Manager}. A critical design decision is the use of \emph{continuous confidence degradation} rather than binary accept/reject logic. Stale readings or those nearing operational boundary limits incur a proportional penalty, preserving marginal information content for downstream Bayesian fusion.

% ---- Reasoning ----
\subsubsection{Reasoning Layer}
\label{sec:reasoning}
The Reasoning layer implements a five-stage pipeline that transforms evidence into justified, confidence-weighted decisions. The architectural decision to employ Bayesian inference \cite{thrun2005probabilistic} as the fusion mechanism addresses the need for principled uncertainty quantification. 

For each hypothesis $h$ with prior $P(h)$ and a set of evidence items $\{e_1, \ldots, e_n\}$, the log-posterior is computed as:
\begin{equation}
  \log P(h \mid \mathbf{e}) = \log P(h) + \sum_{i=1}^{n} c_i \cdot \log \Lambda_i(h)
  \label{eq:bayesian_fusion}
\end{equation}
where $c_i \in [0,1]$ is the confidence of evidence item $e_i$ and $\Lambda_i(h)$ is the likelihood ratio of $e_i$ for hypothesis $h$. The confidence-weighted exponentiation ensures that low-confidence evidence contributes proportionally less to the posterior update. 

Crucially, a \texttt{ReasoningTrace} object accumulates the complete provenance chain: all generated hypotheses, retrieved knowledge constraints, Bayesian likelihood updates, and the final selected decision. This granular evidence provenance ensures end-to-end explainability for field operations.

% ---- Planning & Learning ----
\subsubsection{Planning and Learning Layers}
\label{sec:planning_learning}
The Planning layer translates decisions into executable task graphs (DAGs) composed of registered skills, ensuring plans can be inspected before physical commitment. The Learning layer closes the cognitive loop by comparing expected and actual outcomes through quantitative prediction error analysis. It updates Semantic Memory using an exponential moving average (EMA):
\begin{equation}
  \theta_{t+1} = (1 - \alpha) \cdot \theta_t + \alpha \cdot x_{\mathrm{observed}}
  \label{eq:ema_update}
\end{equation}
where $\alpha$ is a configurable learning rate and $x_{\mathrm{observed}}$ is the actual observed field value.

% ---- Meta-Cognition ----
\subsubsection{Meta-Cognition Layer}
\label{sec:meta_cognition}
The Meta-Cognition layer provides self-monitoring capabilities. It assesses the confidence propagation map, detects competing hypotheses (evaluating probability gaps and Shannon entropy), and escalates to human operators via a priority-ordered cascade when autonomous confidence falls below safety thresholds.

% ============================================================
\subsection{Memory Architecture}
\label{sec:memory}

ACA employs a four-part memory architecture inspired by cognitive science models of human memory \cite{tulving1972episodic, baddeley1992working}. The decision to partition memory reflects the fundamental observation that agricultural knowledge possesses qualitatively different temporal horizons and mutability requirements.

\begin{table}[htbp]
\centering
\caption{Memory subsystem comparison: temporal role, mutability, and access pattern.}
\label{tab:memory_subsystems}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{p{2.0cm}p{3.5cm}p{4.0cm}p{3.5cm}}
\toprule
\textbf{Memory} & \textbf{Cognitive Analogy} & \textbf{Mutability / Bounds} & \textbf{Access Pattern} \\
\midrule
Working & Scratchpad \cite{baddeley1992working} & R/W/D, Bounded (FIFO) & Namespace + key \\
Episodic & Event journal \cite{tulving1972episodic} & Append-only, Unbounded & Zone-indexed multi-filter \\
Semantic & Knowledge base & Read/write, Freezable & Domain + key \\
Farm & Asset registry & R/W (no delete), Unbounded & ID + zone-relational \\
\bottomrule
\end{tabular}%
}
\end{table}

% ============================================================
\subsection{Knowledge and World Representation}
\label{sec:knowledge_world}

\subsubsection{Knowledge Layer and Agentic RAG}
The Knowledge Layer provides external agronomic expertise to the cognitive pipeline. The architecture adopts an Agentic Retrieval-Augmented Generation (RAG) framework \cite{lewis2020rag} rather than statically embedding logic. The layer is defined by an \texttt{AbstractEmbedder} and an \texttt{AbstractVectorStore} that index and retrieve knowledge chunks via cosine similarity. 

An Agronomy Knowledge Tool autonomously queries this vector store, generating dense representations of field anomalies, retrieving relevant treatment constraints, and assigning a Granular Evidence Provenance Score based on vector similarity and source authority. This ensures the Bayesian fusion engine in the Reasoning layer can properly weight retrieved agronomic literature against live sensor telemetry.

\subsubsection{World Model and Digital Twin}
The World Model maintains a dynamic property graph representation of the physical farm environment, capturing relational structures (e.g., zones containing sensors). The Digital Twin provides a strict predictive simulation environment \cite{jones2003dssat} decoupled from the live state, executing deterministic crop trajectory simulations. Natural resource depletion (e.g., moisture and nitrogen) is modelled via exponential decay:
\begin{equation}
  m_t = m_{t-1} \cdot (1 - r_{\mathrm{evap}}), \quad
  n_t = n_{t-1} \cdot (1 - r_{\mathrm{leach}})
  \label{eq:decay}
\end{equation}

% ============================================================
\subsection{Tool--Skill Abstraction and Orchestration}
\label{sec:tool_skill_orch}

ACA enforces a strict separation between \emph{tools} (atomic, stateless interactions with the environment) and \emph{skills} (multi-step workflows composing tool invocations). This separation ensures that tools never reason, and skills never directly interact with hardware. 

The orchestration framework coordinates execution via a typed publish--subscribe \texttt{MessageBus}. The Workflow Engine manages the task DAG, identifying ready tasks and delegating them to the Scheduler. The Scheduler applies a pluggable routing policy, distributing heavy computational tasks to cloud runtimes while routing time-sensitive actuation commands to edge devices.

% ============================================================
\subsection{Verification and Reproducibility Strategy}
\label{sec:verification}

The architecture is validated through a comprehensive, milestone-driven verification strategy. Rather than evaluating the system on real-world crop yield—which conflates architectural soundness with agronomic model accuracy—the evaluation targets contract enforcement, mathematically sound evidence fusion, and computational stability.

The implementation relies solely on the Python standard library and NumPy to eliminate complex dependency resolution and supply-chain risk, directly supporting experimental reproducibility. The deterministic discrete-time simulation model within the Digital Twin guarantees byte-identical trajectory predictions given identical initial configurations. Furthermore, the use of proxy gateways validates that agent contracts (e.g., restricted memory access, explicit tool allowlists) successfully prevent scope creep and unauthorised environmental actuation.

% ADDED: This is required to properly link your references.bib file
\bibliographystyle{apalike} % Changed from ieeetr or IEEEtran
\bibliography{references}