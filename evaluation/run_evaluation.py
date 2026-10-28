
import json
from pathlib import Path

from app.retrieval.hybrid_search import HybridSearch, load_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "evaluation_questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "evaluation_results.json"


def validate_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation questions must be a non-empty list.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Question {index} has no valid question text.")

        if not isinstance(item.get("relevant_titles"), list):
            raise ValueError(
                f"Question {index} must have a relevant_titles list."
            )

        if "answerable" in item and not isinstance(item["answerable"], bool):
            raise ValueError(
                f"Question {index}: answerable must be true or false."
            )


def recall_at_k(results, relevant_titles, k):
    relevant_titles = set(relevant_titles)

    if not relevant_titles:
        return None

    found_titles = {item["title"] for item in results[:k]}
    return len(found_titles & relevant_titles) / len(relevant_titles)


def reciprocal_rank_at_k(results, relevant_titles, k):
    relevant_titles = set(relevant_titles)

    if not relevant_titles:
        return None

    for rank, item in enumerate(results[:k], start=1):
        if item["title"] in relevant_titles:
            return 1 / rank

    return 0.0

def detect_no_answer(results, threshold):
    if not results:
        return True

    return results[0]["rerank_score"] < threshold


def evaluate_strategy(strategy, questions):
    print(f"\nLoading chunks: {strategy}", flush=True)
    documents = load_chunks(strategy=strategy)
    print("Chunks loaded:", len(documents), flush=True)

    search_engine = HybridSearch(documents)

    recalls = []
    reciprocal_ranks = []
    details = []

    no_answer_count = 0
    no_answer_correct = 0
    no_answer_false_accepts = 0
    answerable_false_rejections = 0

    # დროებითი ზღვარი — საბოლოოდ უნდა დაკალიბრდეს
    threshold = 2.5

    for index, item in enumerate(questions, start=1):
        query = item["question"]
        relevant_titles = item["relevant_titles"]
        answerable = item.get("answerable", True)

        results = search_engine.search(
            query,
            top_k=10,
            candidate_k=200,
        )

        # No-answer detection
        predicted_no_answer = (
            detect_no_answer(results, threshold)
        )

        if answerable:
            if predicted_no_answer:
                answerable_false_rejections += 1
        else:
            no_answer_count += 1

            if predicted_no_answer:
                no_answer_correct += 1
            else:
                no_answer_false_accepts += 1

        # Retrieval metrics — მხოლოდ პასუხიანი კითხვებისთვის
        if answerable and relevant_titles:
            recall = recall_at_k(
                results,
                relevant_titles,
                5,
            )
            mrr = reciprocal_rank_at_k(
                results,
                relevant_titles,
                10,
            )

            recalls.append(recall)
            reciprocal_ranks.append(mrr)
        else:
            recall = None
            mrr = None

        details.append({
            "question": query,
            "answerable": answerable,
            "relevant_titles": relevant_titles,
            "top_5_titles": [
                result["title"]
                for result in results[:5]
            ],
            "top_rerank_score": (
                results[0]["rerank_score"]
                if results else None
            ),
            "predicted_no_answer": predicted_no_answer,
            "recall_at_5": recall,
            "reciprocal_rank_at_10": mrr,
        })

        print(
            f"{index}/{len(questions)} evaluated: {query}",
            flush=True,
        )

    mean_recall = (
        sum(recalls) / len(recalls)
        if recalls else 0.0
    )

    mean_mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    no_answer_accuracy = (
        no_answer_correct / no_answer_count
        if no_answer_count else 0.0
    )

    print(f"\nStrategy: {strategy}")
    print(f"Answerable questions scored: {len(recalls)}")
    print(
        "No-answer questions excluded from Recall/MRR:",
        no_answer_count,
    )
    print(f"Mean Recall@5: {mean_recall:.3f}")
    print(f"Mean MRR@10: {mean_mrr:.3f}")

    print(
        f"No-answer correctly detected: "
        f"{no_answer_correct}/{no_answer_count}"
    )
    print(
        f"No-answer false accepts: "
        f"{no_answer_false_accepts}/{no_answer_count}"
    )
    print(
        "Answerable questions falsely rejected: "
        f"{answerable_false_rejections}/{len(recalls)}"
    )
    print(
        f"No-answer detection accuracy: "
        f"{no_answer_accuracy:.3f}"
    )

    return {
        "strategy": strategy,
        "chunks_loaded": len(documents),
        "answerable_questions_scored": len(recalls),
        "no_answer_questions_excluded": no_answer_count,
        "mean_recall_at_5": mean_recall,
        "mean_mrr_at_10": mean_mrr,
        "no_answer_correct": no_answer_correct,
        "no_answer_false_accepts": no_answer_false_accepts,
        "answerable_false_rejections": answerable_false_rejections,
        "no_answer_detection_accuracy": no_answer_accuracy,
        "threshold": threshold,
        "question_results": details,
    }

def main():
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)

    validate_questions(questions)
    print("Evaluation questions:", len(questions))

    all_results = []

    for strategy in ("fixed", "sentence"):
        all_results.append(evaluate_strategy(strategy, questions))

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    for strategy_result in all_results:
        print(f"\n--- {strategy_result['strategy']} ---")

        for item in strategy_result["question_results"]:
            if item["answerable"] != (
                    not item["predicted_no_answer"]
            ):
                print("Question:", item["question"])
                print("Expected answerable:", item["answerable"])
                print("Predicted no answer:", item["predicted_no_answer"])
                print("Top rerank score:", item["top_rerank_score"])


if __name__ == "__main__":
    main()
