struct SetupError: Error, CustomStringConvertible {
    let problem: String
    let fix: String

    init(_ problem: String, fix: String) {
        self.problem = problem
        self.fix = fix
    }

    var description: String { "\(problem)\n  Fix: \(fix)" }
}
