# Bash completion for vidflow
# Install: vidflow completion bash --install
# Or: vidflow completion bash > ~/.local/share/bash-completion/completions/vidflow

_vidflow_complete_dirs() {
    local cur="$1"
    compopt -o filenames -o nospace
    mapfile -t COMPREPLY < <(compgen -d -- "$cur")
}

_vidflow_complete_files() {
    local cur="$1"
    local ext="$2"
    compopt -o filenames -o nospace
    mapfile -t COMPREPLY < <(compgen -f -X "!$ext" -- "$cur")
}

_vidflow_complete_files_or_dirs() {
    local cur="$1"
    local ext="$2"
    compopt -o filenames -o nospace
    local files dirs
    mapfile -t files < <(compgen -f -X "!$ext" -- "$cur")
    mapfile -t dirs < <(compgen -d -- "$cur")
    COMPREPLY=("${files[@]}" "${dirs[@]}")
}

_vidflow_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Handle 'completion' subcommand
    if [[ "${COMP_WORDS[1]}" == "completion" ]]; then
        case "$COMP_CWORD" in
            2)
                COMPREPLY=($(compgen -W "bash" -- "$cur"))
                ;;
            *)
                COMPREPLY=($(compgen -W "--install --path" -- "$cur"))
                ;;
        esac
        return 0
    fi

    # Detect active subcommand
    local subcmd=""
    for word in "${COMP_WORDS[@]:1}"; do
        case "$word" in
            youtube|local|transcribe) subcmd="$word"; break ;;
        esac
    done

    # Subcommand completion at position 1
    if [[ -z "$subcmd" ]]; then
        COMPREPLY=($(compgen -W "youtube local transcribe completion" -- "$cur"))
        return 0
    fi

    # Common transcribe flags
    local transcribe_flags="-m --model --batch-size --context-frames --temperature --max-dimension -c --context -t --title -y --yes --dry-run --estimate-only"
    local common_flags="-v --verbose -q --quiet --json -h --help"

    # Flags requiring specific argument completion
    case "$prev" in
        -o|--output)
            _vidflow_complete_dirs "$cur"
            return 0
            ;;
        --frame-format)
            COMPREPLY=($(compgen -W "jpg png" -- "$cur"))
            return 0
            ;;
        -m|--model|--language|-t|--title)
            # User types these manually
            return 0
            ;;
        -c|--context)
            compopt -o filenames -o nospace
            mapfile -t COMPREPLY < <(compgen -f -- "$cur")
            return 0
            ;;
        --interval|--max-frames|--batch-size|--context-frames|--temperature|--max-dimension|--dedup-threshold)
            # Numeric arguments, no completion
            return 0
            ;;
    esac

    # Flag completion when cur starts with -
    if [[ "$cur" == -* ]]; then
        local opts=""
        case "$subcmd" in
            youtube)
                opts="-o --output --interval --max-frames --frame-format --language --prefer-manual --dedup-threshold --no-dedup --keep-video --no-ai-title --transcribe --merge $transcribe_flags $common_flags"
                ;;
            local)
                opts="-o --output --interval --max-frames --frame-format --dedup-threshold --no-dedup --fast --no-fast -f --force --transcribe --merge $transcribe_flags $common_flags"
                ;;
            transcribe)
                opts="-o --output $transcribe_flags $common_flags"
                ;;
        esac
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return 0
    fi

    # Positional argument completion
    case "$subcmd" in
        youtube)
            # URLs are typed manually
            COMPREPLY=()
            ;;
        local)
            _vidflow_complete_files_or_dirs "$cur" "*.@(mp4|mkv|avi|mov|webm|flv|wmv)"
            ;;
        transcribe)
            _vidflow_complete_files_or_dirs "$cur" "*.md"
            ;;
    esac
}

complete -F _vidflow_completions vidflow
