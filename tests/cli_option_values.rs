use pssa::cli::CLIHandler;

#[test]
fn option_like_missing_value_is_rejected_before_typed_parsing() {
    let error = CLIHandler::parse_and_execute(vec![
        "pssa".into(),
        "train".into(),
        "--epochs".into(),
        "--bogus".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "option '--epochs' requires a value");
}

#[test]
fn hf_flags_reject_invalid_or_ambiguous_sources_before_network_access() {
    for (args, expected) in [
        (vec!["--hf-dataset", "bad-name"], "expected an owner/name"),
        (
            vec!["--hf-dataset", "owner/name", "science"],
            "cannot be combined",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--data", "science"],
            "cannot be combined",
        ),
        (vec!["--hf-config", "default"], "require --hf-dataset"),
        (vec!["--hf-split", "train"], "require --hf-dataset"),
        (vec!["--hf-field", "text"], "require --hf-dataset"),
        (
            vec!["--hf-dataset", "owner/name", "--hf-split", ""],
            "must not be empty",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--hf-field", ""],
            "must not be empty",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--hf-config", ""],
            "must not be empty",
        ),
        (vec!["--hf-dataset"], "requires a value"),
    ] {
        let mut command = vec!["pssa".to_string(), "train".to_string()];
        command.extend(args.iter().map(|s| s.to_string()));
        let error = CLIHandler::parse_and_execute(command).unwrap_err();
        assert!(error.contains(expected), "{args:?}: {error}");
    }
}

#[test]
fn cached_hf_training_emits_real_feed_and_matches_local_checkpoint() {
    use pssa::{dataset::Tokenizer, training::sequence_plan};
    use std::hash::{Hash, Hasher};
    use std::{fs, process::Command};

    let root = std::env::temp_dir().join(format!("pssa-hf-cli-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    // Small, offline cache fixture in /tmp. Keep this key in sync with the
    // versioned on-disk cache contract; no live dataset is needed by tests.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    "hf-row-lines-v2".hash(&mut hash);
    "fixture/corpus".hash(&mut hash);
    Some("tiny".to_string()).hash(&mut hash);
    "train".hash(&mut hash);
    "text".hash(&mut hash);
    let raw = "one two three four five six seven eight nine ten\nshort row\na different sized final sample with extra tokens\n";
    fs::write(root.join(format!("{:016x}.txt", hash.finish())), raw).unwrap();
    let local = root.join("local.txt");
    fs::write(&local, raw).unwrap();
    let run = |hf: bool| {
        let checkpoint = root.join(if hf { "hf.pssa" } else { "local.pssa" });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pssa"));
        command.arg("train");
        if hf {
            command.args(["--hf-dataset", "fixture/corpus", "--hf-config", "tiny"]);
        } else {
            command.arg(&local);
        }
        let output = command
            .args([
                "--tokenizer",
                "word",
                "--latent",
                "8",
                "--state",
                "2",
                "--key",
                "2",
                "--memory",
                "4",
                "--chunk",
                "3",
                "--accumulate",
                "2",
                "--batch-size",
                "2",
                "--skip-tokens",
                "3",
                "--max-tokens",
                "30",
                "-e",
                "2",
                "--no-tui",
                "-o",
            ])
            .arg(&checkpoint)
            .env("PSSA_HF_CACHE", &root)
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8(output.stdout).unwrap();
        assert!(!log.contains('\x1b'), "--no-tui must stay plain");
        (log, fs::read(checkpoint).unwrap())
    };
    let (feed, hf_checkpoint) = run(true);
    let (plain, local_checkpoint) = run(false);
    assert_eq!(
        hf_checkpoint, local_checkpoint,
        "feed metadata must not affect training bits"
    );
    assert!(
        !root.join("local.txt.pssatok").exists(),
        "omitting --token-cache must not enable the persistent cache"
    );
    assert!(!plain.contains("feed_"));
    let tokenizer = Tokenizer::from_corpus(raw, true).unwrap();
    let docs = CLIHandler::documents(raw, &tokenizer, Some(30), 3).unwrap();
    let plan = sequence_plan(&docs, 3, 2).unwrap();
    let c = plan.iter().flatten().last().unwrap();
    let end = c.start + c.len;
    let ids = docs[c.doc][end.saturating_sub(16).max(c.start)..end]
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("%2C");
    let last = feed
        .lines()
        .filter(|line| line.contains("feed_dataset="))
        .last()
        .unwrap();
    let tokens: usize = plan.iter().flatten().map(|c| c.len).sum::<usize>() * 2;
    for expected in [
        "feed_dataset=fixture/corpus".to_string(),
        format!("feed_rows={}", docs.len() * 2),
        format!("feed_tokens={tokens}"),
        format!("feed_row={}", c.doc + 1),
        format!("feed_token_ids={ids}"),
    ] {
        assert!(
            last.split_whitespace().any(|field| field == expected),
            "missing {expected}: {last}"
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn negative_numeric_option_value_reaches_domain_validation() {
    let error = CLIHandler::parse_and_execute(vec![
        "pssa".into(),
        "generate".into(),
        "prompt".into(),
        "--temperature".into(),
        "-1".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "--temp must be >= 0");
}

#[test]
fn interdiffusion_optimizer_writes_and_resumes_a_normal_checkpoint() {
    use std::{fs, process::Command};

    let root = std::env::temp_dir().join(format!("pssa-interdiffusion-cli-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let corpus = root.join("corpus.txt");
    let first = root.join("first.pssa");
    let second = root.join("second.pssa");
    fs::write(
        &corpus,
        "alpha beta gamma delta epsilon zeta eta theta\n".repeat(8),
    )
    .unwrap();
    let run = |output: &std::path::Path, extra: &[&str]| {
        let mut args = vec![
            "train",
            corpus.to_str().unwrap(),
            "--optimizer",
            "interdiffusion",
            "--tokenizer",
            "word",
            "--epochs",
            "1",
            "--latent",
            "8",
            "--state",
            "2",
            "--key",
            "2",
            "--memory",
            "4",
            "--chunk",
            "4",
            "--batch-size",
            "1",
            "--accumulate",
            "1",
            "--no-tui",
            "--out",
            output.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
            .args(args)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    };
    run(&first, &[]);
    run(&second, &["--resume", first.to_str().unwrap()]);
    assert!(fs::metadata(&first).unwrap().len() > 22);
    assert!(fs::metadata(&second).unwrap().len() > 22);
    fs::remove_dir_all(root).unwrap();
}
