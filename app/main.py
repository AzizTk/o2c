import logging
import mlflow
import json
from dataclasses import asdict
from loaders.json_loaders import load_emails_from_json
from pipeline.classify import classify_email
from pipeline.extract import extract_fields
from pipeline.draft import generate_draft_response
from pipeline.route import route_email

#TODO: fix llm json output, make sure the error doesnt happen
HUMAN_REVIEW_QUEUE = "AR Support"

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ---------- Step 1: Ingest ----------
    logger.info("STEP 1/5 — Ingesting emails")
    emails = load_emails_from_json("app/sample_emails.json")
    logger.info(f"Ingested {len(emails)} emails")

    with mlflow.start_run():
        # ---- Run-level context ----
        mlflow.set_tag("run_type", "email_batch")
        mlflow.log_param("email_count", len(emails))

        confidences = []
   #     queue_counts = {
    #        "Cash Application": 0,
    #        "Disputes": 0,
    #        "AR Support": 0,
    #    }

        # ---------- Process each email ----------
        for idx, email in enumerate(emails, start=1):
            email_id = email.metadata.get("id", "unknown")
            logger.info(f"(Email {idx}/{len(emails)} | {email_id}) START processing")

            try:
                # ---------- Step 2: Classify ----------
                logger.info(f"(Email {idx} | {email_id}) STEP 2/5 — Classifying")
                classification = classify_email(email)
                confidences.append(classification.confidence)
                logger.info(
                    f"(Email {idx} | {email_id}) Classified as "
                    f"{classification.case_type} (confidence={classification.confidence:.2f})"
                )
                mlflow.log_text(
                    classification.raw_llm_output,
                    artifact_file=f"{email_id}/classification.json",
                )
                # ---------- Step 3: Extract ----------
                logger.info(f"(Email {idx} | {email_id}) STEP 3/5 — Extracting fields")
                extracted = extract_fields(email, classification)
                logger.info(
                    f"(Email {idx} | {email_id}) Extracted "
                    f"invoices={extracted.invoice_ids}, amount={extracted.amount}"
                )
                mlflow.log_text(
                    json.dumps(asdict(extracted), indent=2),
                    artifact_file=f"{email_id}/extraction.json",
                )
                # ---------- Step 4: Draft (skipped) ----------
                draft = None

                if classification.confidence >= 0.8: # Needs fine-tuning based on use case
                    logger.info(
                        f"(Email {idx} | {email_id}) STEP 4/5 — Generating draft response"
                    )
                    draft = generate_draft_response(
                        email=email,
                        classification=classification,
                        extracted=extracted,
                    )

                # ---------- Step 5: Route ----------
                logger.info(f"(Email {idx} | {email_id}) STEP 5/5 — Routing")
                queue = route_email(classification)
                status = "processed"
                logger.info(f"(Email {idx} | {email_id}) Routed to queue: {queue}")


                if draft:
                    mlflow.log_text(
                        draft,
                        artifact_file=f"{email_id}/draft.txt",
                    )

            except Exception as e:
                # ---------- Safe failure ----------
                queue = HUMAN_REVIEW_QUEUE
                status = "failed"

                logger.error(
                    f"(Email {idx} | {email_id}) ERROR — {type(e).__name__}: {e}"
                )

                mlflow.log_text(
                    str(e),
                    artifact_file=f"{email_id}/error.txt",
                )

            # ---------- Always log outcome ----------
           # queue_counts[queue] += 1
            mlflow.log_text(
                json.dumps(
                    {
                        "email_id": email_id,
                        "status": status,
                        "queue": queue,
                        "confidence": classification.confidence,
                        "case_type": classification.case_type,
                    },
                    indent=2,
                ),
                artifact_file=f"{email_id}/outcome.json",
            )

            logger.info(f"(Email {idx}/{len(emails)} | {email_id}) END processing\n")


if __name__ == "__main__":
    main()
