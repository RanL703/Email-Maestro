from __future__ import annotations


TASK_SEEDS = {
    "easy_deadline_extraction": {
        "emails": [
            {
                "sender": "prof.smith@university.edu",
                "recipient": "assistant@workspace.local",
                "subject": "Course project milestones",
                "body": (
                    "Please track these deadlines: proposal due 2026-04-10, "
                    "prototype due 2026-04-20, and final report due 2026-04-30."
                ),
                "timestamp": "2026-04-04T09:00:00Z",
            }
        ],
        "files": [],
    },
    "medium_triage_and_negotiation": {
        "emails": [
            {
                "sender": "news@updates.example",
                "recipient": "assistant@workspace.local",
                "subject": "Weekly industry digest",
                "body": "Newsletter content 1",
                "timestamp": "2026-04-04T08:00:00Z",
            },
            {
                "sender": "promotions@vendor.example",
                "recipient": "assistant@workspace.local",
                "subject": "Exclusive offer",
                "body": "Newsletter content 2",
                "timestamp": "2026-04-04T08:05:00Z",
            },
            {
                "sender": "events@community.example",
                "recipient": "assistant@workspace.local",
                "subject": "Upcoming events",
                "body": "Newsletter content 3",
                "timestamp": "2026-04-04T08:10:00Z",
            },
            {
                "sender": "client@company.com",
                "recipient": "assistant@workspace.local",
                "subject": "Urgent: delivery issue",
                "body": "A critical complaint needs escalation.",
                "timestamp": "2026-04-04T08:20:00Z",
            },
            {
                "sender": "teammate@company.com",
                "recipient": "assistant@workspace.local",
                "subject": "Need to reschedule",
                "body": "Can we move our sync? Please propose a new time.",
                "timestamp": "2026-04-04T08:30:00Z",
            },
        ],
        "files": [],
    },
    "hard_rag_reply": {
        "emails": [
            {
                "sender": "vip.stakeholder@company.com",
                "recipient": "assistant@workspace.local",
                "subject": "Need Q3 architecture metrics",
                "body": "Please share the key Q3 architecture metrics from the report.",
                "timestamp": "2026-04-04T07:30:00Z",
            }
        ],
        "files": [
            {
                "filename": "Q3_Architecture_Report.txt",
                "content_text": (
                    "Q3 Architecture Report\n"
                    "System availability: 99.95%\n"
                    "Mean API latency: 182ms\n"
                    "Infrastructure cost reduction: 14%\n"
                ),
            }
        ],
    },
}
