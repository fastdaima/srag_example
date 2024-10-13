from ragatouille import RAGPretrainedModel
import re 
import polars as pl 

# data = pl.read_csv('scrum_glossary.csv')

# map_ = {k:v for k,v in zip(data['glossary'].to_list(), data['link'].to_list())}

map_ = {'5 Whys': None, 'AI (Artificial Intelligence)': 'https://scaledagileframework.com/ai', 'ART Backlog': 'https://scaledagileframework.com/art-and-solution-train-backlogs/', 'ART Flow': 'https://scaledagileframework.com/art-flow/', 'ART Kanban': None, 'ART PI Risks': None, 'ART Planning Board': None, 'ART Predictability Measure': None, 'ART Sync': None, 'Acceptance Criteria': None, 'Acceptance Test Driven Development (ATDD)': None, 'Agile': None, 'Agile Architecture': 'https://scaledagileframework.com/agile-architecture/', 'Agile Business Function': None, 'Agile Business Train': None, 'Agile Executive Team': None, 'Agile Manifesto': None, 'Agile Product Delivery (APD)': 'https://scaledagileframework.com/agile-product-delivery/', 'Agile Release Train (ART)': 'https://scaledagileframework.com/agile-release-train/', 'Agile Software Engineering': None, 'Agile Teams': 'https://scaledagileframework.com/agile-teams/', 'Architect Sync': None, 'Architectural Runway': 'https://scaledagileframework.com/architectural-runway/', 'Backlog Refinement': None, 'Baseline Solution Investments (BSIs)': None, 'Batch Size': None, 'Behavior-Driven Development (BDD)': None, 'Benefit Hypothesis': None, 'Big Data': 'https://scaledagileframework.com/big-data', 'Built-In Quality': 'https://scaledagileframework.com/built-in-quality/', 'Burn-Down (Burn-Up) Chart': None, 'Business Agility': 'https://scaledagileframework.com/business-agility/', 'Business Context': None, 'Business Owners': 'https://scaledagileframework.com/business-owners/', 'Business and Technology': 'https://scaledagileframework.com/business-and-technology', 'Business-Enabled ART': None, 'CALMR': 'https://scaledagileframework.com/calmr', 'Capabilities': 'https://scaledagileframework.com/features-and-capabilities/', 'Capacity Allocation': None, 'Cloud': 'https://scaledagileframework.com/cloud/', 'Coach Sync': None, 'Collective Ownership': None, 'Combined Portfolio': None, 'Communities of Practice (CoPs)': 'https://scaledagileframework.com/communities-of-practice/', 'Compliance': 'https://scaledagileframework.com/compliance/', 'Confidence Vote': None, 'Continuous Delivery Pipeline (CDP)': 'https://scaledagileframework.com/continuous-delivery-pipeline/', 'Continuous Deployment (CD)': 'https://scaledagileframework.com/continuous-deployment/', 'Continuous Exploration (CE)': 'https://scaledagileframework.com/continuous-exploration/', 'Continuous Integration (CI)': 'https://scaledagileframework.com/continuous-integration/', 'Continuous Learning Culture (CLC)': 'https://scaledagileframework.com/continuous-learning-culture/', 'Coordinate and Deliver': 'https://scaledagileframework.com/coordinate-and-deliver/', 'Core Values': 'https://scaledagileframework.com/safe-core-values/', 'Cost of Delay': None, 'Customer': 'https://scaledagileframework.com/customer-centricity/', 'Customer Centricity': 'https://scaledagileframework.com/customer-centricity/', 'Customer Journey Map': None, 'Decentralized Decision-Making': None, 'Definition of Done': None, 'Deploy': None, 'Design Thinking': 'https://scaledagileframework.com/design-thinking/', 'DevOps': 'https://scaledagileframework.com/devops/', 'Develop on Cadence': None, 'Development Value Streams': 'https://scaledagileframework.com/development-value-streams', 'Empathy Map': None, 'Enablers': 'https://scaledagileframework.com/enablers/', 'Enterprise': 'https://scaledagileframework.com/enterprise/', 'Enterprise Architect': 'https://scaledagileframework.com/enterprise-architect/', 'Enterprise Solution Delivery (ESD)': 'https://scaledagileframework.com/enterprise-solution-delivery/', 'Epic Hypothesis Statement': None, 'Epic Owners': 'https://scaledagileframework.com/epic-owner/', 'Epics': 'https://scaledagileframework.com/epic/', 'Essential SAFe': 'https://scaledagileframework.com/essential-safe/', 'Estimating Poker': None, 'Extreme Programming': None, 'Features': 'https://scaledagileframework.com/features-and-capabilities/', 'Flow': None, 'Flow Distribution': None, 'Flow Efficiency': None, 'Flow Load': None, 'Flow Predictability': None, 'Flow Time': None, 'Flow Velocity': None, 'Foundation': None, 'Full SAFe': None, 'Gemba': None, 'Hackathon': None, 'Innovation and Planning Iteration': 'https://scaledagileframework.com/innovation-and-planning-iteration/', 'Inspect and Adapt': 'https://scaledagileframework.com/inspect-and-adapt/', 'Integration Point': None, 'Investment Horizons': None, 'Iteration': 'https://scaledagileframework.com/iterations/', 'Iteration Goals': None, 'Iteration Planning': 'https://scaledagileframework.com/iteration-planning/', 'Iteration Retrospective': 'https://scaledagileframework.com/iteration-retrospective/', 'Iteration Review': 'https://scaledagileframework.com/iteration-review/', 'Large Solution SAFe': 'https://scaledagileframework.com/large-solution/', 'Lean': None, 'Lean Budget Guardrails': 'https://scaledagileframework.com/guardrails/', 'Lean Budgets': 'https://scaledagileframework.com/lean-budgets/', 'Lean Business Case (LBC)': None, 'Lean Governance': None, 'Lean Portfolio Management (LPM)': 'https://scaledagileframework.com/lean-portfolio-management/', 'Lean Quality Management System (Lean QMS)': None, 'Lean User Experience (Lean UX)': 'https://scaledagileframework.com/lean-ux/', 'Lean-Agile Center of Excellence (LACE)': None, 'Lean-Agile Leadership (LAL)': 'https://scaledagileframework.com/lean-agile-leadership/', 'Lean-Agile Mindset': 'https://scaledagileframework.com/lean-agile-mindset/', "Little's Law": None, 'Measure and Grow': 'https://scaledagileframework.com/measure-and-grow/', 'Milestone': None, 'Minimum Marketable Feature (MMF)': None, 'Minimum Viable Product (MVP)': None, 'Model-Based Systems Engineering (MBSE)': 'https://scaledagileframework.com/model-based-systems-engineering/', 'Modified Fibonacci Sequence': None, 'Nonfunctional Requirements (NFRs)': 'https://scaledagileframework.com/nonfunctional-requirements/', 'Objectives and Key Results (OKRs)': 'https://scaledagileframework.com/okrs/', 'Operational Value Streams (OVS)': 'https://scaledagileframework.com/operational-value-streams', 'Organizational Agility (OA)': 'https://scaledagileframework.com/organizational-agility/', 'PI Objectives': 'https://scaledagileframework.com/pi-objectives/', 'PI Planning': 'https://scaledagileframework.com/pi-planning/', 'Pareto Analysis': None, 'Participatory Budgeting (PB)': 'https://scaledagileframework.com/participatory-budgeting', 'Personas': None, 'Phase Gate': None, 'Plan-Do-Check-Adjust (PDCA)': None, 'Planning Interval (PI)': 'https://scaledagileframework.com/planning-interval/', 'Portfolio': 'https://scaledagileframework.com/portfolio/', 'Portfolio Backlog': 'https://scaledagileframework.com/portfolio-backlog/', 'Portfolio Canvas': None, 'Portfolio Flow': 'https://scaledagileframework.com/portfolio-flow/', 'Portfolio Kanban': None, 'Portfolio SAFe': 'https://scaledagileframework.com/portfolio/', 'Portfolio Vision': 'https://scaledagileframework.com/portfolio-vision/', 'Pre-Plan': 'https://scaledagileframework.com/pre-plan/', 'Problem-Solving Workshop': None, 'Product Management': 'https://scaledagileframework.com/product-management/', 'Product Owner (PO)': 'https://scaledagileframework.com/product-owner/', 'Product Owner (PO) Sync': None, 'Refactoring': None, 'Relative Estimation': None, 'Release': None, 'Release Train Engineer (RTE)': 'https://scaledagileframework.com/release-train-engineer', 'Release on Demand': 'https://scaledagileframework.com/release-on-demand/', 'Relentless Improvement': None, 'Roadmap': 'https://scaledagileframework.com/roadmap/', 'SAFe': 'https://scaledagileframework.com/safe/', 'SAFe Big Picture (BP)': None, 'SAFe Implementation Roadmap': 'https://scaledagileframework.com/implementation-roadmap/', 'SAFe Lean Startup Cycle': None, 'SAFe Lean-Agile Principles': 'https://scaledagileframework.com/safe-lean-agile-principles/', 'SAFe Overview': None, 'SAFe Practice Consultants (SPCs)': 'https://scaledagileframework.com/spc/', 'SAFe Scrum': 'https://scaledagileframework.com/safe-scrum/', 'SAFe Team Kanban': 'https://scaledagileframework.com/safe-team-kanban/', 'SAFe for Government': 'https://scaledagileframework.com/Government/', 'SWOT Analysis': None, 'Scrum Master/Team Coach (SM/TC)': 'https://scaledagileframework.com/scrum-master-team-coach/', 'Set-Based Design': 'https://scaledagileframework.com/set-based-design/', 'Shared Services': 'https://scaledagileframework.com/shared-services/', 'Solution': 'https://scaledagileframework.com/solution/', 'Solution Architect': 'https://scaledagileframework.com/solution-architect', 'Solution Context': 'https://scaledagileframework.com/solution-context/', 'Solution Demo': 'https://scaledagileframework.com/solution-demo/', 'Solution Intent': 'https://scaledagileframework.com/solution-intent/', 'Solution Management': 'https://scaledagileframework.com/solution-management/', 'Solution Train': 'https://scaledagileframework.com/solution-train/', 'Solution Train Backlog': 'https://scaledagileframework.com/art-and-solution-train-backlogs/', 'Solution Train Engineer (STE)': 'https://scaledagileframework.com/solution-train-engineer/', 'Solution Train Flow': 'https://scaledagileframework.com/solution-train-flow/', 'Solution Vision': 'https://scaledagileframework.com/vision/', 'Spanning Palette': None, 'Spike': None, 'Sprint': None, 'Stories': 'https://scaledagileframework.com/story/', 'Story Map': None, 'Story Point': None, 'Strategic Themes': 'https://scaledagileframework.com/strategic-themes/', 'Sunk Costs': None, 'Supplier': 'https://scaledagileframework.com/supplier/', 'System Architect': 'https://scaledagileframework.com/system-architect/', 'System Demo': 'https://scaledagileframework.com/system-demo/', 'System Team': 'https://scaledagileframework.com/system-team/', 'Systems Thinking': None, 'TOWS Analysis': None, 'Team Backlog': 'https://scaledagileframework.com/team-backlog/', 'Team Flow': 'https://scaledagileframework.com/team-flow', 'Team Sync': None, 'Team Topologies': None, 'Team and Technical Agility (TTA)': 'https://scaledagileframework.com/team-and-technical-agility/', 'Test-Driven Development (TDD)': None, 'U-curve Optimization': None, 'Value Management Office (VMO)': None, 'Value Stream': None, 'Value Stream Coordination': 'https://scaledagileframework.com/value-stream-coordination/', 'Value Stream Identification': None, 'Value Stream KPIs': 'https://scaledagileframework.com/value-stream-kpis/', 'Value Stream Management (VSM)': 'https://scaledagileframework.com/value-stream-management/', 'Value Stream Mapping': None, 'Value Streamlet': None, 'Verification and validation (V&V)': None, 'Weighted Shortest Job First (WSJF)': 'https://scaledagileframework.com/wsjf/', 'Work in Process': None}

questions = ['Can You Elaborate ART SYNC?',
'Could you explain ART SYNC in more detail?',
'Can you elaborate on what ART SYNC is?',
'What exactly is ART SYNC?',
'Could you tell me more about the features of ART SYNC?',
'How does ART SYNC work?',
'What are the benefits of using ART SYNC?',
"What you mean by ART SYNC. Can you clarify?",
"I'm a bit confused about ART SYNC. Could you elaborate?",
'Can you explain the concept of ART SYNC in simpler terms?',
'Can you tell me about ART SYNC?',
'What does ART SYNC do?',
'Could you explain what ART SYNC is?',
"I'm not familiar with ART SYNC. Can you give me a brief overview?",
'What is the purpose of ART SYNC?',
'How is ART SYNC typically used or implemented?',
'What are some real-world examples of ART SYNC in action?']

document_ids = {str(ind_ + 1) : o for ind_,o in enumerate(map_.keys())}

def create_rag_index(): 
    rag = RAGPretrainedModel.from_pretrained(
        'colbert-ir/colbertv2.0'
    )
    print('map data', len(map_))


    rag.index(
        collection=list(map_.keys()),
        document_ids=list(document_ids.keys()),
        index_name='scrum_glossary',
        max_document_length=len(map_),
        split_documents=False 
    )

def query_index(rag, query):
    docs = rag.search(query,)
    return docs 
    


import os
from groq import Groq

client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def call_groq(subset, query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Return the corresponding links from {subset} for the terms present in {query} in json format queried term: link",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

if __name__ == '__main__':
    # to create indices
    # create_rag_index() 
    path = '.ragatouille/colbert/indexes/scrum_glossary'
    rag = RAGPretrainedModel.from_index(path) 

    out_file = [] 

    for q in questions:
        out_file.append('\n\n')
        out_file.append(q)

        docs = query_index(rag, q)
        subset = ['\n\n']
        
        for o in docs:
            key = document_ids[o['document_id']]
            value = map_[key]
            if value: subset.append(f'{key}:{value}')
        subset = '\n'.join(subset)
        out_file.append(subset)

    with open('out.txt', 'w') as f: 
        f.writelines(out_file)