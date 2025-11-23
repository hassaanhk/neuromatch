import streamlit as st
import os
import json
import itertools
from groq import Groq
from typing import List, Dict

# --- Configuration & Initialization ---
st.set_page_config(page_title="NeuroMatch 🧠", layout="wide")

# FIX: Retrieve API Key directly from environment variable (Deployment Secret)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = None
LLM_MODEL = "llama-3.1-8b-instant"

# Initialize client cleanly
try:
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    client = None

# If client failed to initialize, show warning
if client is None and GROQ_API_KEY:
    st.warning("LLM client could not be fully initialized. Check your Groq Key and internet connection.")
elif client is None and not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set. Running in mock/fallback mode. Please set the API key in deployment secrets.")


# --- Utility: Safe LLM call with parsing and fallback ---

def call_llm(system_prompt: str, user_content: str) -> dict:
    """Call the Groq LLM (if configured). Returns a parsed JSON dict or an error object."""
    if client is None:
        # Simplified fallback for mock/no-key mode
        return {"error": "No GROQ client configured. Running in mock mode."}

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )

        content = response.choices[0].message.content

        # Robust JSON parsing
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                # Fallback parser to strip potential noise
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(content[start:end+1])
                return {"error": "LLM returned non-parsable JSON content", "raw": content}

    except Exception as e:
        return {"error": str(e)}

    return {"error": "Unknown LLM error occurred."}


# --- System Prompts ---
PROFILE_SYSTEM_PROMPT = """
You are NeuroMatch — an AI specialized in analyzing human thinking styles, decision patterns, and collaboration traits.
Analyze the provided participant answers and output a JSON object exactly matching the structure below. Be concise, specific, and avoid generic platitudes.
{
  "name": "",
  "archetype": "",
  "summary": "",
  "strengths": [],
  "blind_spots": [],
  "working_style": "",
  "ideal_collaboration_partner": "",
  "communication_style": "",
  "stress_behavior": "",
  "motivation_drivers": [],
  "risk_tendencies": "",
  "keywords": []
}
"""

COMPATIBILITY_SYSTEM_PROMPT = """
You are NeuroMatch Pairwise Analyzer. You receive TWO cognitive profiles in JSON.
Compute an integer compatibility_score between 1 and 100 (higher = better collaboration potential).
Also output a short collaboration_style, combined_strengths (list), potential_conflicts (list), and 2-4 practical recommendations.
Return exactly a JSON object like:
{
  "pair_name": "A vs B",
  "compatibility_score": 0,
  "collaboration_style": "",
  "combined_strengths": [],
  "potential_conflicts": [],
  "recommendations": []
}
"""

# FIX: Team Map Prompt Updated for Visualization Fields
TEAM_MAP_SYSTEM_PROMPT = """
You are the NeuroMatch Team Synthesizer. Given a compressed list of team profiles and summarized pairwise compatibility scores, generate a final team summary in JSON following this structure exactly:
{
  "team_overview": "Summary of the overall team cognitive balance using markdown.",
  "archetype_distribution": [],
  "team_strengths": [],
  "team_weaknesses": [],
  "risk_factors": [],
  "recommended_team_structure": "Suggest roles and leadership for **every team member** using markdown.",
  "workflow_recommendation": "Suggest the best project flow using markdown.",
  "performance_prediction": "One-line prediction of team performance in a project setting using markdown and an emoji."
}
Keep text clear, actionable, and reference members by name when suggesting roles. The 'archetype_distribution' field should be a list of dictionaries: [{'archetype': 'Strategist', 'count': 2}].
"""

# --- Questions ---
QUESTIONS = [
"1. How do you approach a difficult problem?",
"2. Do you prefer planning or experimenting?",
"3. How do you make decisions under pressure?",
"4. Do you focus on details or the big picture?",
"5. What slows you down when working with others?",
"6. What makes you feel confident during a task?",
"7. How fast do you respond when new information appears?",
"8. Do you rely on logic or intuition?",
"9. Describe your communication style in one sentence.",
"10. What role do you naturally take in a group?",
]

# --- Session State Initialization ---
if 'profiles' not in st.session_state:
    st.session_state.profiles = []
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'compatibility_results' not in st.session_state:
    st.session_state.compatibility_results = []
if 'team_map' not in st.session_state:
    st.session_state.team_map = None
if 'current_profile' not in st.session_state:
    st.session_state.current_profile = None


# FIX: Remove st.rerun() from here to prevent infinite loops
def set_page(page_name: str):
    st.session_state.page = page_name


# --- Core Functions (Fallbacks remain the same) ---

def generate_profile_from_answers(name: str, answers: Dict[str, str]) -> Dict:
    """Format user answers and call LLM to generate a profile, with fallback."""
    payload = f"Participant Name: {name}\nAnswers:\n"
    for i, q in enumerate(QUESTIONS):
        payload += f"{q} {answers.get(f'q{i+1}', '')}\n"

    result = call_llm(PROFILE_SYSTEM_PROMPT, payload)

    if 'error' in result:
        # Fallback heuristic:
        combined = ' '.join(answers.values())
        archetype = 'Analyst' if 'detail' in combined or 'plan' in combined else 'Executor'
        return {
            "name": name,
            "archetype": archetype,
            "summary": f"(Fallback) Heuristic profile generated due to LLM error: {result['error']}",
            "strengths": ["reliable"],
            "blind_spots": ["may be terse"],
            "working_style": "structured",
            "ideal_collaboration_partner": "Complementary thinker",
            "communication_style": "clear",
            "stress_behavior": "focuses inward",
            "motivation_drivers": ["completion"],
            "risk_tendencies": "moderate",
            "keywords": [k for k in combined.split()[:10]]
        }
    return result


def heuristic_compatibility_score(arche_a: str, arche_b: str) -> int:
    """Simple heuristic to estimate compatibility."""
    if not arche_a or not arche_b: return 50
    if arche_a == arche_b: return 65
    pairs_good = [
        ('visionary','analyst'), ('visionary','executor'), ('strategist','executor'),
        ('connector','analyst'), ('creative solver','analyst')
    ]
    if (arche_a.lower(), arche_b.lower()) in pairs_good or (arche_b.lower(), arche_a.lower()) in pairs_good:
        return 85
    return 60


def generate_compatibility(profiles: List[Dict]) -> List[Dict]:
    """Generates pairwise compatibility results."""
    results = []
    pairs = list(itertools.combinations(profiles, 2))

    for a, b in pairs:
        pair_name = f"{a.get('name','A')} vs {b.get('name','B')}"
        user_content = f"Compare Profile A and Profile B:\nProfile A: {json.dumps(a)}\nProfile B: {json.dumps(b)}"

        comparison = call_llm(COMPATIBILITY_SYSTEM_PROMPT, user_content)

        if 'error' in comparison:
            # Fallback heuristic
            arche_a = a.get('archetype','')
            arche_b = b.get('archetype','')
            score = heuristic_compatibility_score(arche_a, arche_b)
            comparison = {
                'pair_name': pair_name,
                'compatibility_score': score,
                'collaboration_style': f'Heuristic-Estimate (LLM Error: {comparison["error"][:20]}...)',
                'combined_strengths': list(set(a.get('strengths', []) + b.get('strengths', [])))[:5],
                'potential_conflicts': list(set(a.get('blind_spots', []) + b.get('blind_spots', [])))[:5],
                'recommendations': ["Communicate expectations clearly", "Define roles upfront"]
            }

        if isinstance(comparison.get('compatibility_score'), (int, float)):
            comparison['compatibility_score'] = int(comparison['compatibility_score'])
        else:
            comparison['compatibility_score'] = int(comparison.get('compatibility_score', 50))

        comparison['name_a'] = a.get('name')
        comparison['name_b'] = b.get('name')
        results.append(comparison)

    st.session_state.compatibility_results = results
    return results


def generate_team_map():
    """Generate a concise team brain map."""
    if not st.session_state.profiles:
        st.error("No profiles available for team mapping.")
        return None

    # Check if compatibility was generated, if not, generate it now (safety)
    if not st.session_state.compatibility_results:
        generate_compatibility(st.session_state.profiles)

    # Compress profiles to essential fields
    compressed_profiles = []
    for p in st.session_state.profiles:
        compressed_profiles.append({
            'name': p.get('name'),
            'archetype': p.get('archetype'),
            'strengths': p.get('strengths', [])[:6],
            'blind_spots': p.get('blind_spots', [])[:6],
        })

    compressed_compat = []
    for c in st.session_state.compatibility_results:
        compressed_compat.append({
            'pair': c.get('pair_name'),
            'score': c.get('compatibility_score'),
            'collab': c.get('collaboration_style')
        })

    payload = json.dumps({
        'team_size': len(st.session_state.profiles),
        'profiles': compressed_profiles,
        'compatibility': compressed_compat
    })

    result = call_llm(TEAM_MAP_SYSTEM_PROMPT, payload)

    if 'error' in result:
        # Fallback team map
        names = [p.get('name') for p in st.session_state.profiles]
        fallback = {
            'team_overview': f"Team of {len(names)} members: {' ,'.join(names)}. (Fallback due to LLM error: {result['error']})",
            'archetype_distribution': [{'archetype': p.get('archetype'), 'count': 1} for p in st.session_state.profiles],
            'team_strengths': ['Diverse thinking', 'Complementary skills'],
            'team_weaknesses': ['Potential coordination overhead'],
            'risk_factors': ['Unaligned priorities under stress'],
            'recommended_team_structure': '\n'.join([f"- {p.get('name')}: {p.get('archetype')}" for p in st.session_state.profiles]),
            'workflow_recommendation': 'Define sprints, assign clear owners, and run daily syncs.',
            'performance_prediction': 'Moderate to high if communication is prioritized. 🚀'
        }
        st.session_state.team_map = fallback
        return fallback

    st.session_state.team_map = result
    return result


# --- UI Pages ---

def display_welcome_page():
    st.title("🎤 NeuroMatch: Team Cognitive Analyzer")
    st.write("Welcome! This tool analyzes your team's thinking styles to generate a 'Team Brain Map'.")
    st.markdown("---")
    st.header(f"Team Size: **{len(st.session_state.profiles)}** members assessed.")
    # FIX: Add st.rerun() inside the button block
    if st.button("Start Assessment"):
        set_page('questionnaire')
        st.rerun() 


def display_questionnaire():
    st.title(f"Assessment for Team Member {len(st.session_state.profiles) + 1}")
    st.caption("Answer candidly. Your response will be used to generate a cognitive profile.")
    st.markdown('--')

    # FIX: Use a unique key for the form using the number of profiles
    with st.form(key=f'form_{len(st.session_state.profiles)}'):
        name = st.text_input("Your Name/Alias", max_chars=40)
        answers = {}
        for i, q in enumerate(QUESTIONS):
            answers[f'q{i+1}'] = st.text_area(q, height=80, key=f'ans_{len(st.session_state.profiles)}_{i}')

        submitted = st.form_submit_button("Generate My Profile")

        if submitted:
            if not name.strip():
                st.error("Please enter your name/alias.")
            else:
                with st.spinner(f"Analyzing {name}... (may take a few seconds)"):
                    profile = generate_profile_from_answers(name, answers)

                if profile:
                    # Success/Fallback
                    st.success("Profile generated.")
                    st.session_state.profiles.append(profile)
                    # Clear compatibility and team map cache upon new profile added
                    st.session_state.compatibility_results = []
                    st.session_state.team_map = None
                    set_page('profile_review')
                    st.rerun() # FIX: Rerun after profile is added
                else:
                    st.error("Failed to create profile. Try again.")


def display_profile_review():
    st.title("✅ Team Profile Review")
    st.header(f"Total Members Assessed: {len(st.session_state.profiles)}")
    st.markdown('--')

    # FIX: Ensure all profiles display well
    for idx, profile in enumerate(st.session_state.profiles, start=1):
        with st.expander(f"{idx}. {profile.get('name','N/A')} — The {profile.get('archetype','N/A')}"):
            st.write(profile.get('summary',''))
            st.write('**Strengths:**', ', '.join(profile.get('strengths', [])))
            st.write('**Blind Spots:**', ', '.join(profile.get('blind_spots', [])))
            st.write('**Ideal Partner:**', profile.get('ideal_collaboration_partner', 'N/A'))

    col1, col2 = st.columns(2)
    with col1:
        # FIX: Add st.rerun()
        if st.button(f"Start Assessment for Member {len(st.session_state.profiles) + 1}"):
            set_page('questionnaire')
            st.rerun() 
    with col2:
        if len(st.session_state.profiles) >= 2:
            # FIX: Add st.rerun()
            if st.button("Analyze Team Compatibility 🚀"):
                set_page('compatibility_matrix')
                st.rerun()


def display_compatibility_matrix():
    st.title("🛡️ Compatibility Matrix & Conflict Zones")

    if len(st.session_state.profiles) < 2:
        st.warning("You need at least two members to run compatibility analysis.")
        if st.button("Back to Welcome"):
            set_page('welcome')
            st.rerun()
        return

    # Check 1: If no results, show button to generate them
    if not st.session_state.compatibility_results:
        st.info("Compatibility results are not generated yet. Click the button below to analyze all pairs.")
        if st.button("Start Pairwise Analysis Now"):
            with st.spinner("Calculating compatibility scores..."):
                generate_compatibility(st.session_state.profiles)
                st.rerun() # FIX: Use st.rerun()
        return

    # Check 2: If results exist, display them
    st.header(f"Total Profiles Analyzed: {len(st.session_state.profiles)}")
    st.markdown('--')

    for res in st.session_state.compatibility_results:
        score = res.get('compatibility_score', 0)
        if score >= 80:
            icon = '✅'
        elif score >= 60:
            icon = '⚠️'
        else:
            icon = '🛑'

        with st.expander(f"{icon} {res.get('pair_name')} — Score: **{score}/100**"):
            st.markdown(f"**Collaboration Style:** {res.get('collaboration_style','N/A')}")
            st.write('**Combined Strengths:**', ', '.join(res.get('combined_strengths', [])))
            st.write('**Potential Conflicts:**', ', '.join(res.get('potential_conflicts', [])))
            st.write('**Recommendations:**', '; '.join(res.get('recommendations', [])))

    st.markdown('--')
    if st.button("Generate Final Team Brain Map"):
        with st.spinner('Synthesizing team map...'):
            generate_team_map()
            set_page('team_brain_map')
            st.rerun() # FIX: Rerun after generating map


def display_team_map():
    st.title('🧠 Final Team Brain Map')

    # Safety Check: If user navigates here without generating the map
    if not st.session_state.team_map:
        st.warning('Team map not generated yet. Generating map now...')
        with st.spinner('Synthesizing team map...'):
             generate_team_map()
        st.rerun()
        return

    tm = st.session_state.team_map

    # Calculate Average Score for Metrics
    avg_score = round(sum(c['compatibility_score'] for c in st.session_state.compatibility_results) / len(st.session_state.compatibility_results), 1) if st.session_state.compatibility_results else 0

    st.header('Performance Prediction')
    st.markdown(tm.get('performance_prediction', 'N/A'))

    st.markdown("---")

    # FIX: ADVANCED VISUALIZATION using st.columns and st.metric
    st.markdown("## 📊 Team Composition Snapshot")
    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Total Members", len(st.session_state.profiles))
    col_b.metric("Team Synergy Score", f"{avg_score}/100", help="Average pairwise compatibility score.")

    # Display top three archetypes (assuming LLM returns archetype_distribution)
    if 'archetype_distribution' in tm and isinstance(tm['archetype_distribution'], list):
        top_archetypes = sorted(tm['archetype_distribution'], key=lambda x: x.get('count', 0), reverse=True)
        if top_archetypes:
            col_c.metric(f"Most Common Archetype", f"{top_archetypes[0]['archetype']} ({top_archetypes[0]['count']})")

    st.markdown("---")


    st.subheader('Team Overview')
    st.markdown(tm.get('team_overview', 'N/A'))

    st.subheader('Team Strengths')
    st.write(', '.join(tm.get('team_strengths', [])))

    st.subheader('Team Weaknesses')
    st.write(', '.join(tm.get('team_weaknesses', [])))

    st.subheader('Risk Factors')
    st.write(', '.join(tm.get('risk_factors', [])))

    st.subheader('Recommended Team Structure')
    st.markdown(tm.get('recommended_team_structure', 'N/A'))

    st.subheader('Workflow Recommendation')
    st.markdown(tm.get('workflow_recommendation', 'N/A'))

    st.markdown("---")
    
    # FIX: Add a button to go back to questionnaire without resetting data
    col_nav_a, col_nav_b = st.columns(2)
    
    with col_nav_a:
        if st.button('➕ Add More Members'):
            # Clear analysis cache, but keep profiles
            st.session_state.compatibility_results = []
            st.session_state.team_map = None
            set_page('questionnaire')
            st.rerun()

    with col_nav_b:
        if st.button('🔄 Restart Analysis (Clear All Data)'):
            # Reset session state completely
            st.session_state.profiles = []
            st.session_state.compatibility_results = []
            st.session_state.team_map = None
            set_page('welcome')
            st.rerun()


# --- Main Controller ---

def main_controller():
    st.sidebar.title('NeuroMatch Flow')
    # Simple navigation mapping
    page_map = {
        'Welcome': 'welcome',
        'Questionnaire': 'questionnaire',
        'Profile Review': 'profile_review',
        'Compatibility Matrix': 'compatibility_matrix',
        'Team Brain Map': 'team_brain_map'
    }

    # Determine the index for the radio button based on current page
    current_page_title = st.session_state.page.replace('_',' ').title()
    try:
        current_index = list(page_map.keys()).index(current_page_title)
    except ValueError:
        current_index = 0 # Default to Welcome

    page_selection = st.sidebar.radio('Current Step:',
                                       list(page_map.keys()),
                                       index=current_index,
                                       key='flow_radio')

    # Update page state based on radio selection
    # NOTE: We rely on the radio widget to handle the state change and Streamlit's implicit rerun
    
    
    # Page router
    if st.session_state.page == 'welcome':
        display_welcome_page()
    elif st.session_state.page == 'questionnaire':
        display_questionnaire()
    elif st.session_state.page == 'profile_review':
        display_profile_review()
    elif st.session_state.page == 'compatibility_matrix':
        display_compatibility_matrix()
    elif st.session_state.page == 'team_brain_map':
        display_team_map()


if __name__ == '__main__':
    main_controller()