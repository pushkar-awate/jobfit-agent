"""Live job board with AI fit-scoring (jobfit-agent web demo).

Pulls real job postings from a public feed, scores each one against your resume,
ranks them by an F1-style match strength (so a detailed role you match well beats
a thin posting that happens to list two skills you have), shows what you match and
what you're missing, and links straight to the posting.
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
THIN_POSTING = 2   # postings listing this few skills carry little signal


def _default_resume():
    try:
        with open(os.path.join(_HERE, "examples", "sample_resume.txt"),
                  encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def score_job(job, resume_skills):
    """Match strength of one posting against the resume.

    Returns (match_score, coverage, matched, missing, jd_total).
    match_score is an F1 blend of:
      recall    = matched / jd_total     (how much of the role's ask you cover)
      precision = matched / resume_total (how central the role is to your skills)
    F1 avoids the old trap where a thin posting you fully match outranked a rich,
    highly-relevant one. Sorting uses (match_score, matched_count).
    """
    jd = set(extract_skills(job["match_text"]))
    matched = sorted(jd & resume_skills)
    missing = sorted(jd - resume_skills)
    nm, njd, nr = len(matched), len(jd), len(resume_skills)
    if nm == 0 or njd == 0 or nr == 0:
        return 0, 0, matched, missing, njd
    recall = nm / njd
    precision = nm / nr
    f1 = 2 * recall * precision / (recall + precision)
    return round(100 * f1), round(100 * recall), matched, missing, njd


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
            score, coverage, matched, missing, jd_total = score_job(j, resume_skills)
            ranked.append((score, len(matched), coverage, matched, missing, jd_total, j))
        # rank by match strength, then by absolute number of skills matched
        ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)

    if live:
        st.caption("Your skills: " + (", ".join(sorted(resume_skills)) or "none detected"))
        st.success("Showing %d live roles for \"%s\", ranked by overall match strength."
                   % (len(ranked), query))
    else:
        st.info("The live feed wasn't reachable just now, so these are example "
                "roles - the scoring and ranking work exactly the same.")
    st.caption("Match strength balances how much of a role's ask you cover against "
               "how central it is to your skill set, so a detailed role you match "
               "well ranks above a thin posting that lists a couple of skills.")

    for score, nmatched, coverage, matched, missing, jd_total, j in ranked:
        st.divider()
        top = st.columns([4, 1])
        title = j["title"] or "Role"
        company = j["company"] or "Company"
        top[0].markdown("### %s\n**%s** · %s" %
                        (title, company, j["location"] or "Remote"))
        top[1].metric("Match", "%d" % score,
                      help="0-100 match strength (F1 of coverage and relevance). "
                           "Used to rank roles.")
        st.progress(coverage / 100 if jd_total else 0.0)
        if jd_total <= THIN_POSTING:
            st.caption("⚠️ Limited info - this posting lists only %d skill(s), "
                       "so the match is a rough guess." % jd_total)
        elif matched:
            st.markdown("**You match %d of %d skills this role lists** (%d%% of its asks): %s"
                        % (nmatched, jd_total, coverage, ", ".join(matched)))
        else:
            st.caption("No overlapping skills detected for this posting.")
        if missing:
            st.markdown("**Missing:** " + ", ".join(missing[:8]))
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
