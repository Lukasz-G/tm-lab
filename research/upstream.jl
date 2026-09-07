# Shared helper: make sure the upstream implementations research/ measures against are present.
#
# They are NOT vendored. `reference/` is gitignored, so a fresh clone has nothing there; this
# clones on demand so every experiment stays reproducible without carrying someone else's source
# in our history. See reference/README.md.

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const REFERENCE = joinpath(REPO_ROOT, "reference")

const UPSTREAMS = Dict(
    "BooBSD-FuzzyPatternTM" => "https://github.com/BooBSD/FuzzyPatternTM.git",
    "BooBSD-Tsetlin.jl"     => "https://github.com/BooBSD/Tsetlin.jl.git",
)

"""
    upstream(name) -> path

Absolute path to an upstream checkout under `reference/`, cloning it if absent.
"""
function upstream(name::AbstractString)
    haskey(UPSTREAMS, name) || error("unknown upstream $name; known: $(collect(keys(UPSTREAMS)))")
    path = joinpath(REFERENCE, name)
    if !isdir(path)
        mkpath(REFERENCE)
        @info "cloning upstream" name url = UPSTREAMS[name]
        run(`git clone --depth 50 $(UPSTREAMS[name]) $path`)
    end
    return path
end

"Report the commit an upstream is pinned at, so results can be traced to a source version."
function upstream_commit(name::AbstractString)
    path = upstream(name)
    try
        return strip(read(`git -C $path log -1 --format=%h\ %ad --date=short`, String))
    catch
        return "unknown"
    end
end
