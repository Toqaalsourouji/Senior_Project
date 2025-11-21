#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <numeric>

using namespace std;

struct Task {
    string name;
    int period;
    int execution_time;
    int remaining_time;
    int next_deadline;
    int priority; // For RM: lower period = higher priority (lower number)
    
    Task(string n, int p, int e) : name(n), period(p), execution_time(e), 
                                     remaining_time(0), next_deadline(p), priority(p) {}
};

// Calculate GCD
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// Calculate LCM of two numbers
int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

// Calculate hyperperiod (LCM of all periods)
int calculate_hyperperiod(const vector<Task>& tasks) {
    int hp = tasks[0].period;
    for (size_t i = 1; i < tasks.size(); i++) {
        hp = lcm(hp, tasks[i].period);
    }
    return hp;
}

// Read tasks from file
vector<Task> read_tasks(const string& filename) {
    vector<Task> tasks;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }
    
    string line;
    while (getline(file, line)) {
        // Remove spaces
        line.erase(remove(line.begin(), line.end(), ' '), line.end());
        
        if (line.empty()) continue;
        
        // Parse: T1,6,2
        stringstream ss(line);
        string name, period_str, exec_str;
        
        getline(ss, name, ',');
        getline(ss, period_str, ',');
        getline(ss, exec_str, ',');
        
        int period = stoi(period_str);
        int exec_time = stoi(exec_str);
        
        tasks.push_back(Task(name, period, exec_time));
    }
    
    file.close();
    return tasks;
}

// Check if task set is schedulable under RM
bool check_rm_schedulability(const vector<Task>& tasks) {
    double utilization = 0.0;
    for (const auto& task : tasks) {
        utilization += (double)task.execution_time / task.period;
    }
    
    int n = tasks.size();
    double bound = n * (pow(2.0, 1.0/n) - 1);
    
    cout << "Total Utilization: " << utilization << endl;
    cout << "RM Schedulability Bound: " << bound << endl;
    
    return utilization <= 1.0;
}

// Rate Monotonic Scheduling
void simulate_rm(vector<Task> tasks, int hyperperiod) {
    cout << "\n=== Rate Monotonic (RM) Scheduling ===" << endl;
    cout << "Hyperperiod: " << hyperperiod << endl << endl;
    
    // Sort by period (ascending) - shorter period = higher priority
    sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
        return a.period < b.period;
    });
    
    cout << "Task Priorities (highest to lowest):" << endl;
    for (const auto& task : tasks) {
        cout << task.name << " (Period: " << task.period << ")" << endl;
    }
    cout << endl;
    
    check_rm_schedulability(tasks);
    cout << "\nSchedule:" << endl;
    
    for (int time = 0; time < hyperperiod; time++) {
        // Check for task arrivals (release)
        for (auto& task : tasks) {
            if (time % task.period == 0) {
                task.remaining_time = task.execution_time;
                task.next_deadline = time + task.period;
            }
        }
        
        // Select highest priority task that has work to do
        Task* selected = nullptr;
        for (auto& task : tasks) {
            if (task.remaining_time > 0) {
                selected = &task;
                break; // Already sorted by priority
            }
        }
        
        if (selected != nullptr) {
            cout << "Time " << time << ": " << selected->name << endl;
            selected->remaining_time--;
        } else {
            cout << "Time " << time << ": IDLE" << endl;
        }
    }
}

// Earliest Deadline First Scheduling
void simulate_edf(vector<Task> tasks, int hyperperiod) {
    cout << "\n=== Earliest Deadline First (EDF) Scheduling ===" << endl;
    cout << "Hyperperiod: " << hyperperiod << endl << endl;
    
    // Check schedulability
    double utilization = 0.0;
    for (const auto& task : tasks) {
        utilization += (double)task.execution_time / task.period;
    }
    cout << "Total Utilization: " << utilization << endl;
    cout << "EDF Schedulable: " << (utilization <= 1.0 ? "Yes" : "No") << endl;
    cout << "\nSchedule:" << endl;
    
    for (int time = 0; time < hyperperiod; time++) {
        // Check for task arrivals (release)
        for (auto& task : tasks) {
            if (time % task.period == 0) {
                task.remaining_time = task.execution_time;
                task.next_deadline = time + task.period;
            }
        }
        
        // Select task with earliest deadline that has work to do
        Task* selected = nullptr;
        int earliest_deadline = hyperperiod + 1;
        
        for (auto& task : tasks) {
            if (task.remaining_time > 0 && task.next_deadline < earliest_deadline) {
                earliest_deadline = task.next_deadline;
                selected = &task;
            }
        }
        
        if (selected != nullptr) {
            cout << "Time " << time << ": " << selected->name 
                 << " (Deadline: " << selected->next_deadline << ")" << endl;
            selected->remaining_time--;
        } else {
            cout << "Time " << time << ": IDLE" << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <algorithm>" << endl;
        cout << "Algorithm: RM or EDF" << endl;
        cout << "Example: " << argv[0] << " tasks.txt RM" << endl;
        return 1;
    }
    
    string filename = argv[1];
    string algorithm = argv[2];
    
    // Convert algorithm to uppercase
    transform(algorithm.begin(), algorithm.end(), algorithm.begin(), ::toupper);
    
    // Read tasks from file
    vector<Task> tasks = read_tasks(filename);
    
    if (tasks.empty()) {
        cerr << "Error: No tasks found in file" << endl;
        return 1;
    }
    
    cout << "Tasks loaded:" << endl;
    for (const auto& task : tasks) {
        cout << task.name << ": Period=" << task.period 
             << ", Execution=" << task.execution_time << endl;
    }
    cout << endl;
    
    // Calculate hyperperiod
    int hyperperiod = calculate_hyperperiod(tasks);
    
    // Run selected algorithm
    if (algorithm == "RM") {
        simulate_rm(tasks, hyperperiod);
    } else if (algorithm == "EDF") {
        simulate_edf(tasks, hyperperiod);
    } else {
        cerr << "Error: Unknown algorithm '" << algorithm << "'" << endl;
        cerr << "Please use 'RM' or 'EDF'" << endl;
        return 1;
    }
    
    return 0;
}