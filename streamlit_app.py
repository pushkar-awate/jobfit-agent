"""Live job board with AI fit-scoring (jobfit-agent web demo).

Pulls real job postings from a public feed, scores each one against your resume
using the same skill matcher that powers the jobfit-agent CLI, ranks them by fit,
shows what you match and what you're missing, and links straight to the posting.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.jobs import fetch_jobs, sample_jobs, JobFetchError
from app.skills import extract_skills

st.set_page_config(page_title="jobfit-agent - live job board", page_icon="\U0001F9ED",
                   layout="centered")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _default_resume():
    try:
        with open(os.path.join(_HERE, "examples", "sample_resume.txt"),
                  encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def score_job(job, resume_skills):
    """Keyword fit of one posting against the resume's skills."""
    jd = set(extract_skills(job["match_text"]))
    matched = sorted(jd & resume_skills)
    missing = sorted(jd - resume_skills)
    score = round(100 * len(matched) / max(1, len(jd)))
    return score, matched, missing


st.title("\U0001F9ED jobfit-agent - live job board")
st.markdown(
    "Paste your resume, pick a search, and this pulls **real, live job postings** "
    "and ranks them by how well they fit *you* - using the same skill matcher that "
    "powers the jobfit-agent CLI. Each role shows what you match, what you're "
    "missing, and a link straight to the application."
)

resume_text = st.text_area(
    "Your resume (skills, experience - plain text is fine)",
    value=_default_resume(), height=180,
    help="A sample resume is pre-filled. Replace it with yours to rank jobs for you.")

c1, c2 = st.columns([3, 1])
query = c1.text_input("Search roles", value="machine learning")
limit = c2.slider("How many", 5, 30, 15)

if st.button("Find & rank jobs", type="primary"):
    if not resume_text.strip():
        st.warning("Add your resume above first, so roles can be scored against you.")
        st.stop()

    resume_skills = set(extract_skills(resume_text))
    with st.spinner("Fetching live postings and scoring them against your resume..."):
        live = True
        try:
            jobs = fetch_jobs(query, limit=limit)
            if not jobs:
                live = False
                jobs = sample_jobs()
        except JobFetchError:
            live = False
            jobs = sample_jobs()

        ranked = []
        for j in jobs:
            score, matched, missing = score_job(j, resume_skills)
            ranked.append((score, matched, missing, j))
        ranked.sort(key=lambda r: r[0], reverse=True)

    if live:
        st.caption("Your skills: " + (", ".join(sorted(resume_skills)) or "none detected"))
        st.success("Showing %d live roles for \"%s\", ranked by your fit." %
                   (len(ranked), query))
    else:
        st.info("The live feed wasn't reachable just now, so these are example "
                "roles - the scoring and ranking work exactly the same.")

    for score, matched, missing, j in ranked:
        st.divider()
        top = st.columns([4, 1])
        title = j["title"] or "Role"
        company = j["company"] or "Company"
        top[0].markdown("### %s\n**%s** · %s" %
                        (title, company, j["location"] or "Remote"))
        top[1].metric("Fit", "%d%%" % score)
        st.progress(score / 100)
        if matched:
            st.markdown("**You match:** " + ", ".join(matched))
        if missing:
            st.markdown("**Missing:** " + ", ".join(missing[:8]))
        if not matched and not missing:
            st.caption("No overlapping skills detected for this posting.")
        bottom = st.columns([1, 3])
        if j.get("url"):
            bottom[0].link_button("Apply →", j["url"])
        if j.get("published"):
            bottom[1].caption("Posted %s" % j["published"])
        if j.get("description"):
            with st.expander("Job description"):
                st.write(j["description"][:1500] +
                         ("..." if len(j["description"]) > 1500 else ""))

st.divider()
st.caption("Live feed: Remotive (free, real remote roles). Scoring reuses the "
           "jobfit-agent skill matcher. Source: github.com/pushkar-awate/jobfit-agent")
