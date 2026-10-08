# Bash completion for vidflow
# Install: vidflow completion bash --install
# Or: vidflow completion bash > ~/.local/share/bash-completion/completions/vidflow
#
# The per-subcommand option lists below are checked against the argparse
# parser by tests/test_completion.py; update both together.

_vidflow_complete_dirs() {
    local cur="$1"
    compopt -o filenames -o nospace
    mapfile -t COMPREPLY < <(compgen -d -- "$cur")
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

    local subcommands="youtube local transcribe polish completion"

    # Top level: subcommand names, or top-level options after '-'
    if [[ "$COMP_CWORD" -eq 1 ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=($(compgen -W "-h --help --version" -- "$cur"))
        else
            COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
        fi
        return 0
    fi

    local subcmd="${COMP_WORDS[1]}"

    if [[ "$subcmd" == "completion" ]]; then
        if [[ "$COMP_CWORD" -eq 2 ]]; then
            COMPREPLY=($(compgen -W "bash" -- "$cur"))
        else
            COMPREPLY=($(compgen -W "--install --path" -- "$cur"))
        fi
        return 0
    fi

    local youtube_opts="-h --help -o --output --interval --max-frames --frame-format --language --prefer-manual --dedup-threshold --no-dedup --keep-video --no-ai-title -f --force --transcribe --polish -m --model --provider --temperature --batch-size --context-frames --max-dimension -c --context -t --title -y --yes --dry-run --estimate-only --keep-capture -v --verbose -q --quiet --json"
    local local_opts="-h --help -o --output --interval --max-frames --frame-format --dedup-threshold --no-dedup --fast --no-fast -f --force --no-subtitles --subtitle-track --list-subtitles --transcribe --polish --merge -m --model --provider --temperature --batch-size --context-frames --max-dimension -c --context -t --title -y --yes --dry-run --estimate-only --keep-capture -v --verbose -q --quiet --json"
    local transcribe_opts="-h --help -o --output --merge -m --model --provider --temperature --batch-size --context-frames --max-dimension -c --context -t --title -y --yes --dry-run --estimate-only --keep-capture -v --verbose -q --quiet --json"
    local polish_opts="-h --help -o --output -m --model --provider --temperature --batch-size --context-frames -c --context -y --yes --dry-run --estimate-only --keep-capture -v --verbose -q --quiet --json"

    local opts=""
    case "$subcmd" in
        youtube) opts="$youtube_opts" ;;
        local) opts="$local_opts" ;;
        transcribe) opts="$transcribe_opts" ;;
        polish) opts="$polish_opts" ;;
        *) return 0 ;;
    esac

    # Values for options that take an argument
    case "$prev" in
        -o|--output)
            case "$subcmd" in
                transcribe|polish) _vidflow_complete_files_or_dirs "$cur" "*.md" ;;
                *) _vidflow_complete_dirs "$cur" ;;
            esac
            return 0
            ;;
        -c|--context)
            _vidflow_complete_files_or_dirs "$cur" "*.md"
            return 0
            ;;
        --frame-format)
            COMPREPLY=($(compgen -W "jpg png" -- "$cur"))
            return 0
            ;;
        --provider)
            COMPREPLY=($(compgen -W "local anthropic" -- "$cur"))
            return 0
            ;;
        -m|--model|--language|-t|--title|--interval|--max-frames|--dedup-threshold|--subtitle-track|--temperature|--batch-size|--context-frames|--max-dimension)
            # Free-form or numeric; typed manually
            return 0
            ;;
    esac

    # Options after '-', and for youtube on an empty word too, since its
    # positionals are URLs that cannot be completed
    if [[ "$cur" == -* || "$subcmd" == "youtube" ]]; then
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return 0
    fi

    # Positional arguments
    case "$subcmd" in
        local)
            _vidflow_complete_files_or_dirs "$cur" "*.@(mp4|m4v|mkv|avi|mov|webm|flv|wmv)"
            ;;
        transcribe|polish)
            _vidflow_complete_files_or_dirs "$cur" "*.md"
            ;;
    esac
}

complete -F _vidflow_completions vidflow
