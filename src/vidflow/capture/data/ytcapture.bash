# Bash completion for ytcapture
# Install: ytcapture completion bash --install
# Or: ytcapture completion bash > ~/.local/share/bash-completion/completions/ytcapture

_ytcapture_completions() {
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

    # Flags requiring directory argument
    case "$prev" in
        -o|--output)
            compopt -o filenames -o nospace
            mapfile -t COMPREPLY < <(compgen -d -- "$cur")
            return 0
            ;;
        --language)
            # No completion for language codes
            return 0
            ;;
        --interval|--max-frames|--dedup-threshold)
            # Numeric arguments, no completion
            return 0
            ;;
        --frame-format)
            COMPREPLY=($(compgen -W "jpg png" -- "$cur"))
            return 0
            ;;
    esac

    # Flag completion
    if [[ "$cur" == -* ]]; then
        local opts="-o --output --interval --max-frames --frame-format --language --prefer-manual --dedup-threshold --no-dedup --keep-video --no-ai-title -y --yes -v --verbose --version -h --help"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return 0
    fi

    # Subcommand completion at position 1
    if [[ "$COMP_CWORD" -eq 1 ]] && [[ "completion" == "$cur"* ]]; then
        COMPREPLY=("completion")
        return 0
    fi

    # Default: no file completion (URLs are typed manually)
    COMPREPLY=()
}

complete -F _ytcapture_completions ytcapture
